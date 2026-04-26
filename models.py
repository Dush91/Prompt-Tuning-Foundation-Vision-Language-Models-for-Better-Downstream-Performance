"""
models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


CLIP_TEMPLATES = [
    "a photo of a {}.", "a bad photo of a {}.", "a photo of many {}.",
    "a sculpture of a {}.", "a photo of the hard to see {}.",
    "a low resolution photo of the {}.", "a rendering of a {}.",
    "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.",
    "a tattoo of a {}.", "the embroidered {}.", "a photo of a hard to see {}.",
    "a bright photo of a {}.", "a photo of a clean {}.", "a photo of a dirty {}.",
    "a dark photo of the {}.", "a drawing of a {}.", "a photo of my {}.",
    "the plastic {}.", "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.",
    "a bright photo of the {}.", "a cropped photo of a {}.", "a plastic {}.",
    "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.",
    "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.",
    "a doodle of a {}.", "a close-up photo of the {}.", 
    "the origami {}.", "the {} in a video game.", "a sketch of a {}.",
    "a doodle of the {}.", "a origami {}.", "a low resolution photo of a {}.",
    "the toy {}.", "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.",
    "a jpeg corrupted photo of the {}.", "a good photo of a {}.", "a plushie {}.",
    "a photo of the nice {}.", "a photo of the small {}.", "a photo of the weird {}.",
    "the cartoon {}.", "art of the {}.", "a drawing of the {}.",
    "a photo of the large {}.", "a black and white photo of a {}.",
    "the plushie {}.", "a dark photo of a {}.", "itap of a {}.",
    "graffiti of the {}.", "a toy {}.", "itap of my {}.",
    "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}.",
]
LLM_PROMPTS = {
    'oxford_pets': [
        "a photo of a {}, a type of pet.",
        "a close-up photo of a {} breed.",
        "a {} sitting or standing.",
        "a cute {} animal.",
        "a photo of a {} dog or cat.",
        "a professional photo of a {}.",
        "a {} pet looking at the camera.",
    ],
    'eurosat': [
        "a satellite image of {}.",
        "an aerial photo of {}.",
        "a remote sensing image of {}.",
        "a top-down view of {} land.",
        "a Sentinel-2 satellite image of {}.",
        "an overhead image showing {} area.",
        "a land use photo of {}.",
    ],
    'dtd': [
        "a photo of a {} texture.",
        "a {} surface.",
        "a {} material up close.",
        "a close-up of {} texture.",
        "a photo showing {} pattern.",
        "a {} surface texture.",
        "a detailed view of {} material.",
    ]
}


# ── Zero-shot ─────────────────
@torch.no_grad()
def build_zeroshot_weights(clip_model, tokenizer, class_names, templates, device):
    clip_model.eval()
    weights = []
    for cls in class_names:
        texts  = [t.format(cls) for t in templates]
        tokens = tokenizer(texts).to(device)
        with torch.amp.autocast('cuda'):
            embs = clip_model.encode_text(tokens).float()
        embs = F.normalize(embs, dim=-1)
        weights.append(F.normalize(embs.mean(0), dim=0))
    return torch.stack(weights, dim=1)   # (D, C)

# ── CoCoOp ──────────────
class MetaNet(nn.Module):
    def __init__(self, img_dim, ctx_dim, hidden_ratio=0.125):
        super().__init__()
        hidden = max(1, int(img_dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ctx_dim),
        )
    def forward(self, x): return self.net(x)


class CoCoOp(nn.Module):
    def __init__(self, clip_model, class_names: List[str],
                 n_ctx: int = 16, hidden_ratio: float = 0.125):
        super().__init__()
        self.add_module('clip', clip_model)
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx

        for p in self.clip.parameters():
            p.requires_grad_(False)

        import clip
        with torch.no_grad():
            embedding = clip_model.token_embedding

        ctx_dim = embedding.weight.shape[1]
        img_dim = clip_model.visual.output_dim

        ctx = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx, std=0.02)
        self.ctx      = nn.Parameter(ctx)
        self.meta_net = MetaNet(img_dim, ctx_dim, hidden_ratio)

        with torch.no_grad():
            # 1. Tokenize ONLY the class names (so we don't double up on "a photo of")
            toks = clip.tokenize(class_names).to(embedding.weight.device)
            
            # 2. Shift tokens right by n_ctx to make physical room for the context vectors
            B_toks = toks.shape[0]
            dummy = torch.zeros(B_toks, n_ctx, dtype=toks.dtype, device=toks.device)
            shifted_toks = torch.cat([toks[:, :1], dummy, toks[:, 1 : -n_ctx]], dim=1)
            
            # 3. Find the true EOT position in this properly shifted layout
            eot_pos = shifted_toks.argmax(dim=-1) 
            
            # 4. Generate the embeddings
            embs = embedding(shifted_toks)

        # 5. Register the aligned buffers
        self.register_buffer('eot_positions', eot_pos)  # No need to add n_ctx anymore!
        self.register_buffer('prefix', embs[:, :1, :])  # [SOT] token
        self.register_buffer('suffix', embs[:, 1 + n_ctx:, :]) # [CLASS] + [EOT] + [PADs]

    def _encode_prompts_batch(self, ctx_shifted: torch.Tensor) -> torch.Tensor:
        """
        Process all (B * n_cls) prompts in ONE transformer call.
        ctx_shifted: (B, n_ctx, ctx_dim)
        returns:     (B, n_cls, D)
        """
        B = ctx_shifted.shape[0]
        C = self.n_cls

        # build prompts: (B, C, seq, D)
        prefix  = self.prefix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, 1, D)
        suffix  = self.suffix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, *, D)
        ctx     = ctx_shifted.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, n_ctx, D)
        prompts = torch.cat([prefix, ctx, suffix], dim=2)          # (B, C, seq, D)

        seq_len = prompts.shape[2]
        flat = prompts.view(B * C, seq_len, -1)                    # (B*C, seq, D)
        flat = flat + self.clip.positional_embedding.unsqueeze(0)
        flat = flat.permute(1, 0, 2)                               # (seq, B*C, D)
        flat = self.clip.transformer(flat)
        flat = flat.permute(1, 0, 2)                               # (B*C, seq, D)
        flat = self.clip.ln_final(flat)

        
        eos  = self.eot_positions.unsqueeze(0).expand(B, -1).reshape(B * C)  # (B*C,)
        feat = flat[torch.arange(B * C, device=flat.device), eos]            # (B*C, D)
        feat = feat @ self.clip.text_projection
        feat = F.normalize(feat, dim=-1)
        return feat.view(B, C, -1)                                            # (B, C, D)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        img_feat = F.normalize(self.clip.encode_image(images), dim=-1)  # (B, D)
        return self.forward_cached(img_feat)

    def forward_cached(self, img_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward from pre-cached, normalised image features.
        Skips encode_image — use when CLIP encoder is frozen and features
        are cached once before training (big speedup over full forward).
        img_feat: (B, D) — already L2-normalised, on GPU
        """
        delta       = self.meta_net(img_feat)                    # (B, ctx_dim)
        ctx_shifted = self.ctx.unsqueeze(0) + delta.unsqueeze(1) # (B, n_ctx, D)
        text_feats  = self._encode_prompts_batch(ctx_shifted)    # (B, C, D)
        logits = (img_feat.unsqueeze(1) * text_feats).sum(-1)
        return logits * self.clip.logit_scale.exp()              # (B, C)
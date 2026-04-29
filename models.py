import torch
import torch.nn as nn
import clip

PROMPTS = {
    "stl10": [
        "a photo of a {}.",
        "a clear photo of a {}.",
        "a close-up photo of a {}.",
        "a natural image of a {}.",
        "a realistic photo of a {}.",
        "a cropped photo of a {}.",
        "a photo showing a {}.",
        "a high quality image of a {}.",
        "a centered photo of a {}.",
        "a well-lit photo of a {}."
    ],

    "eurosat": [
        "a satellite image of {}.",
        "an aerial photo of {}.",
        "a remote sensing image of {}.",
        "a top-down view of {} land.",
        "a Sentinel-2 satellite image of {}.",
        "an overhead image showing {} area.",
        "a land use image of {}.",
        "a satellite view of {} terrain.",
        "a geographic image of {} region.",
        "an earth observation image of {}."
    ],

    "caltech101": [
        "a photo of a {}.",
        "a clear image of a {}.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
    
    ],

    "flowers102": [
        "a photo of a {} flower.",
        "a close-up photo of a {} flower.",
        "a clear image of a {} flower.",
        "a detailed photo of a {} flower.",
        "a blooming {} flower.",
        "a colorful {} flower.",
        "a natural photo of a {} flower.",
        "a high resolution image of a {} flower.",
        "a garden photo of a {} flower.",
        "a macro shot of a {} flower."
        
    ],

    "oxfordpets": [
        "a photo of a {}, a type of pet.",
        "a close-up photo of a {} breed.",
        "a clear image of a {} pet.",
        "a photo of a {} dog or cat.",
        "a {} pet looking at the camera.",
        "a realistic photo of a {}.",
        "a cute photo of a {} pet.",
        "a high resolution image of a {}.",
        "a centered photo of a {} animal.",
        "a well-lit photo of a {} pet."
    ]
}

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_final(x).type(self.dtype)

        eos_positions = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_positions] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx, dataset_name, device):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        self.dataset_name = dataset_name
        self.num_templates = len(PROMPTS[dataset_name])

        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompts = []

        for name in classnames:
            for template in PROMPTS[dataset_name]:
                prompts.append("X " * n_ctx + template.format(name))

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        ).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).float()

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(len(self.tokenized_prompts), -1, -1)

        prompts = torch.cat(
            [
                self.token_prefix,
                ctx,
                self.token_suffix
            ],
            dim=1
        )

        return prompts
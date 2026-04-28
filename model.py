# model.py — LoRA layer definition and injection into CLIP ViT attention blocks

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wraps a frozen Linear layer and adds a low-rank side path:
        output = xW + scale * x(AB)
    A is random init, B is zero init → A@B = 0 at start (LoRA paper).
    Only A and B are trainable.
    """
    def __init__(self, linear, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.linear  = linear
        self.scale   = alpha / rank
        d_in, d_out  = linear.in_features, linear.out_features
        self.lora_A  = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B  = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + self.scale * F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B
        )


def _make_forward(attn_block):
    """Patched forward for CLIP MultiheadAttention using LoRA Q and V."""
    def forward(q, k, v, **kwargs):
        return F.multi_head_attention_forward(
            q, k, v,
            embed_dim_to_check = attn_block.embed_dim,
            num_heads          = attn_block.num_heads,
            in_proj_weight     = None,
            in_proj_bias       = None,
            bias_k             = attn_block.bias_k,
            bias_v             = attn_block.bias_v,
            add_zero_attn      = attn_block.add_zero_attn,
            dropout_p          = 0.0,
            out_proj_weight    = attn_block.out_proj.weight,
            out_proj_bias      = attn_block.out_proj.bias,
            training           = attn_block.training,
            q_proj_weight      = attn_block.lora_q.linear.weight + attn_block.lora_q.scale * (attn_block.lora_q.lora_B @ attn_block.lora_q.lora_A),
            k_proj_weight      = attn_block.wk,
            v_proj_weight      = attn_block.lora_v.linear.weight + attn_block.lora_v.scale * (attn_block.lora_v.lora_B @ attn_block.lora_v.lora_A),
            use_separate_proj_weight = True,
            need_weights       = kwargs.get("need_weights", False),
            attn_mask          = kwargs.get("attn_mask"),
        )
    return forward


def inject_lora(model, rank=4, alpha=1.0, dropout=0.0, device="cuda"):
    """
    Injects LoRA into Q and V projections of every vision encoder block.
    Freezes original Q and V weights — only A and B matrices train.

    Vision encoder (ViT-B/16): 12 blocks, d_model=768
        A: rank x 768, B: 768 x rank — per Q and V
        Total: 4 x rank x 768 x 12 = 147,456 (rank=4)
    """
    for block in model.visual.transformer.resblocks:
        d = block.attn.embed_dim
        W = block.attn.in_proj_weight.data
        b = block.attn.in_proj_bias.data
        Wq, Wk, Wv = W.chunk(3, dim=0)
        bq, bk, bv = b.chunk(3, dim=0)

        # Create linear layers from original Q and V weights
        linQ = nn.Linear(d, d).to(device)
        linV = nn.Linear(d, d).to(device)
        with torch.no_grad():
            linQ.weight.copy_(Wq); linQ.bias.copy_(bq)
            linV.weight.copy_(Wv); linV.bias.copy_(bv)

        # Freeze original Q and V — only A and B will train
        linQ.weight.requires_grad_(False); linQ.bias.requires_grad_(False)
        linV.weight.requires_grad_(False); linV.bias.requires_grad_(False)

        block.attn.lora_q = LoRALinear(linQ, rank, alpha, dropout).to(device)
        block.attn.lora_v = LoRALinear(linV, rank, alpha, dropout).to(device)
        block.attn.register_buffer("wk", Wk.clone())
        block.attn.register_buffer("bk", bk.clone())
        block.attn.forward = _make_forward(block.attn)

"""Masked multi-head cross-attention architecture for Step 6 demonstrations."""

from __future__ import annotations

import torch
from torch import nn


class MaskedCrossModalFusion(nn.Module):
    def __init__(self, embedding_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.cross_attention=nn.MultiheadAttention(embedding_dim,num_heads,batch_first=True)
        self.gate=nn.Sequential(nn.Linear(embedding_dim*2,embedding_dim),nn.Sigmoid())
        self.layer_norm=nn.LayerNorm(embedding_dim)

    def forward(self,e_st:torch.Tensor,e_se:torch.Tensor,allowed_mask:torch.Tensor):
        if e_st.ndim==2: e_st=e_st.unsqueeze(0)
        if e_se.ndim==2: e_se=e_se.unsqueeze(0)
        if allowed_mask.shape!=(e_st.shape[1],e_se.shape[1]): raise ValueError("Mask shape does not match query/key lengths")
        if not allowed_mask.any(dim=1).all(): raise ValueError("Every query must have at least one allowed semantic key")
        blocked_mask=~allowed_mask.bool()
        context,weights=self.cross_attention(e_st,e_se,e_se,attn_mask=blocked_mask,need_weights=True,average_attn_weights=False)
        gate=self.gate(torch.cat([e_st,context],dim=-1)); joint=self.layer_norm(e_st+gate*context)
        return context.squeeze(0),joint.squeeze(0),weights.squeeze(0),gate.squeeze(0)


"""Small relation-aware graph encoder without PyG dependencies."""

from __future__ import annotations

import torch
from torch import nn


class RelationalGraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int):
        super().__init__()
        self.hidden_dim=hidden_dim; self.num_relations=num_relations
        self.self_1=nn.Linear(input_dim,hidden_dim); self.rel_1=nn.Parameter(torch.empty(num_relations,input_dim,hidden_dim))
        self.self_2=nn.Linear(hidden_dim,hidden_dim); self.rel_2=nn.Parameter(torch.empty(num_relations,hidden_dim,hidden_dim))
        self.relation_decoder=nn.Embedding(num_relations,hidden_dim); self.attribute_decoder=nn.Linear(hidden_dim,input_dim)
        nn.init.xavier_uniform_(self.rel_1); nn.init.xavier_uniform_(self.rel_2); nn.init.xavier_uniform_(self.relation_decoder.weight)

    @staticmethod
    def _layer(x,edge_index,edge_type,self_linear,relation_weights):
        src,dst=edge_index; messages=torch.bmm(x[src].unsqueeze(1),relation_weights[edge_type]).squeeze(1)
        aggregate=torch.zeros(x.shape[0],messages.shape[1],device=x.device,dtype=x.dtype); aggregate.index_add_(0,dst,messages)
        degree=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype); degree.index_add_(0,dst,torch.ones_like(dst,dtype=x.dtype))
        return self_linear(x)+aggregate/degree.clamp_min(1).unsqueeze(1)

    def forward(self,x,edge_index,edge_type):
        h=torch.relu(self._layer(x,edge_index,edge_type,self.self_1,self.rel_1))
        return self._layer(h,edge_index,edge_type,self.self_2,self.rel_2)

    def edge_logits(self,z,edge_index,edge_type):
        src,dst=edge_index; return (z[src]*self.relation_decoder(edge_type)*z[dst]).sum(1)

    def reconstruct_attributes(self,z): return self.attribute_decoder(z)


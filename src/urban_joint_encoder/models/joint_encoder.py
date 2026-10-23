"""Trainable full-dataset joint encoder with a frozen Step 4 branch."""

from __future__ import annotations

import torch
from torch import nn

from urban_joint_encoder.models.cross_attention import MaskedCrossModalFusion
from urban_joint_encoder.models.semantic_encoder import RelationalGraphEncoder
from urban_joint_encoder.models.st_encoder import SpatialTemporalEncoder


class JointSpatiotemporalSemanticEncoder(nn.Module):
    def __init__(self,adjacency:torch.Tensor,st_state_dict:dict[str,torch.Tensor],hidden_dim:int=32,num_relations:int=6):
        super().__init__(); self.hidden_dim=hidden_dim
        self.st_encoder=SpatialTemporalEncoder(adjacency,hidden_dim)
        encoder_state={k.removeprefix("encoder."):v for k,v in st_state_dict.items() if k.startswith("encoder.")}
        self.st_encoder.load_state_dict(encoder_state)
        for parameter in self.st_encoder.parameters(): parameter.requires_grad=False
        self.semantic_encoder=RelationalGraphEncoder(18,hidden_dim,num_relations)
        self.fusion=MaskedCrossModalFusion(hidden_dim,4)
        self.joint_prediction_head=nn.Linear(hidden_dim,1)

    def train(self,mode:bool=True):
        super().train(mode); self.st_encoder.eval(); return self

    def forward(self,x:torch.Tensor,snapshots:list):
        with torch.no_grad(): e_st=self.st_encoder(x)
        joints=[]; semantic_embeddings=[]; attention_weights=[]; contexts=[]
        for batch_index,snapshot in enumerate(snapshots):
            e_se=self.semantic_encoder(snapshot.node_features,snapshot.edge_index,snapshot.edge_type)
            context,joint,weights,_=self.fusion(e_st[batch_index],e_se,snapshot.attention_mask)
            joints.append(joint); semantic_embeddings.append(e_se); attention_weights.append(weights); contexts.append(context)
        e_joint=torch.stack(joints); prediction=self.joint_prediction_head(e_joint).squeeze(-1)
        return {"prediction":prediction,"E_st":e_st,"E_se":semantic_embeddings,"semantic_context":torch.stack(contexts),"E_joint":e_joint,"attention_weights":attention_weights}


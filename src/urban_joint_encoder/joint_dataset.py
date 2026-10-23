"""Temporal samples paired with on-demand causal semantic snapshots."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class JointTrafficDataset(Dataset):
    def __init__(self,normalized_traffic:np.ndarray,target_indices:list[int]|np.ndarray,sequence_length:int=12):
        self.data=torch.tensor(normalized_traffic,dtype=torch.float32); self.target_indices=np.asarray(target_indices,dtype=np.int64); self.sequence_length=sequence_length
        if len(np.unique(self.target_indices))!=len(self.target_indices): raise ValueError("Duplicate target windows are not allowed")
        if self.target_indices.min()<sequence_length or self.target_indices.max()>=len(self.data): raise ValueError("Invalid target indices")
    def __len__(self): return len(self.target_indices)
    def __getitem__(self,index):
        target=int(self.target_indices[index]); return self.data[target-self.sequence_length:target].unsqueeze(-1),self.data[target],target


def joint_collate(batch):
    x=torch.stack([item[0] for item in batch]); y=torch.stack([item[1] for item in batch]); rows=[int(item[2]) for item in batch]; return x,y,rows

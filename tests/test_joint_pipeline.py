from pathlib import Path
import unittest

import numpy as np
import torch

from urban_joint_encoder.causal_semantic_snapshots import CausalSemanticSnapshotBuilder
from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.data_loader import load_traffic_graph
from urban_joint_encoder.models.joint_encoder import JointSpatiotemporalSemanticEncoder


class JointPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=find_project_root(Path.cwd()); cls.builder=CausalSemanticSnapshotBuilder(cls.root,recent_event_horizon=36,max_events=48)

    def test_snapshot_is_causal_and_every_grid_has_key(self):
        snapshot=self.builder.build_semantic_snapshot(3000); audit=self.builder.leakage_audit(snapshot)
        self.assertEqual(snapshot.observation_cutoff,2999); self.assertTrue(audit["evidence_before_target"]); self.assertTrue(audit["ongoing_events_hide_future_end"]); self.assertTrue(snapshot.attention_mask.any(dim=1).all())

    def test_two_times_have_distinct_dynamic_snapshots(self):
        first=self.builder.build_semantic_snapshot(3000); second=self.builder.build_semantic_snapshot(3007)
        signatures=lambda s:[(d["node_id"],d["state"],d["evidence_row_max"]) for d in s.dynamic_metadata]
        self.assertNotEqual(signatures(first),signatures(second))

    def test_joint_forward_and_cross_attention_gradient(self):
        _,adj,_=load_traffic_graph(self.root); checkpoint=torch.load(self.root/"outputs/models/st_encoder_best.pt",map_location="cpu",weights_only=True)
        model=JointSpatiotemporalSemanticEncoder(torch.from_numpy(adj),checkpoint["model_state_dict"]); snapshot=self.builder.build_semantic_snapshot(3000)
        output=model(torch.zeros(1,12,265,1),[snapshot]); loss=output["prediction"].square().mean(); loss.backward(); grad=model.fusion.cross_attention.in_proj_weight.grad
        self.assertEqual(tuple(output["E_joint"].shape),(1,265,32)); self.assertIsNotNone(grad); self.assertGreater(float(grad.abs().sum()),0.0)


if __name__=="__main__": unittest.main()

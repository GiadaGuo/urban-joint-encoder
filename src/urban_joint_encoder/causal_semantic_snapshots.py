"""Causal, incremental semantic snapshots for full-dataset joint training.

The expensive seasonal anomaly signals and static graph are computed once. A
snapshot at target row t exposes observations only through t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from urban_joint_encoder.data_audit import POI_NAMES
from urban_joint_encoder.data_loader import load_traffic_graph
from urban_joint_encoder.semantic_pipeline import RELATIONS


@dataclass
class SemanticSnapshot:
    target_row_index: int
    observation_cutoff: int
    node_ids: list[str]
    node_types: list[str]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    attention_mask: torch.Tensor
    dynamic_metadata: list[dict[str, object]]

    def to(self, device: torch.device | str) -> "SemanticSnapshot":
        return SemanticSnapshot(
            self.target_row_index, self.observation_cutoff, self.node_ids, self.node_types,
            self.node_features.to(device), self.edge_index.to(device), self.edge_type.to(device),
            self.attention_mask.to(device), self.dynamic_metadata,
        )


class CausalSemanticSnapshotBuilder:
    """Build variable-size semantic graphs without materializing NetworkX graphs."""

    feature_dim=18
    slots_per_day=288

    def __init__(self, root: Path, recent_event_horizon: int = 36, max_events: int = 48):
        self.root=root.resolve(); self.traffic,self.adjacency,_=load_traffic_graph(self.root)
        base=self.root/"data/raw/Main data/prediction_data/POI"
        self.poi=np.vstack([pd.read_csv(base/f"poi{i}.csv",header=None,encoding="utf-8-sig").to_numpy(np.float32).ravel() for i in range(1,8)]).T
        self.relation_to_index={name:i for i,name in enumerate(RELATIONS)}
        self.recent_event_horizon=int(recent_event_horizon); self.max_events=int(max_events)
        self._build_static_cache(); start=time.perf_counter(); self.flags,self.scores,self.medians,self.q10=self._precompute_causal_signals(); self.signal_precompute_seconds=time.perf_counter()-start

    def _build_static_cache(self):
        self.base_node_ids=[f"grid:{i}" for i in range(265)]+[f"poi_category:{i}" for i in range(7)]+["rule:monitor_adjacent","rule:review_poi_context"]
        self.base_node_types=["SpatialGrid"]*265+["POICategory"]*7+["GovernanceRule"]*2
        features=np.zeros((274,self.feature_dim),dtype=np.float32); features[:265,0]=1
        maxima=np.maximum(self.poi.max(0),1); features[:265,4:11]=np.log1p(self.poi)/np.log1p(maxima)
        features[265:272,1]=1
        for i in range(7): features[265+i,11+i]=1
        features[272:,3]=1; features[272,15:17]=[1,1]; features[273,[15,17]]=[1,1]
        edges=[]; types=[]
        for grid in range(265):
            for cat,count in enumerate(self.poi[grid]):
                if count>0: edges.append((grid,265+cat)); types.append(self.relation_to_index["HAS_POI"])
        for i,j in zip(*np.where(self.adjacency>0)):
            edges.append((int(i),int(j))); types.append(self.relation_to_index["ADJACENT_TO"])
        self.static_features=features; self.static_edge_index=np.asarray(edges,dtype=np.int64).T; self.static_edge_type=np.asarray(types,dtype=np.int64)

    def _precompute_causal_signals(self):
        flags=np.zeros_like(self.traffic,dtype=bool); scores=np.zeros_like(self.traffic,dtype=np.float32); medians=np.zeros_like(self.traffic,dtype=np.float32); q10=np.zeros_like(self.traffic,dtype=np.float32)
        for row in range(7*self.slots_per_day,len(self.traffic)):
            day=row//self.slots_per_day; slot=row%self.slots_per_day; days=np.arange(max(0,day-21),day)
            indices=np.concatenate([days*self.slots_per_day+((slot+d)%self.slots_per_day) for d in (-1,0,1)])
            reference=self.traffic[indices]; median=np.median(reference,axis=0); threshold=np.quantile(reference,.1,axis=0); mad=np.median(np.abs(reference-median),axis=0)
            score=(median-self.traffic[row])/(1.4826*mad+.25); flag=(self.traffic[row]<=threshold)&(score>=2)&((median-self.traffic[row])>=2)
            flags[row]=flag; scores[row]=score; medians[row]=median; q10[row]=threshold
        return flags,scores,medians,q10

    def _events_at_cutoff(self, cutoff:int):
        start=max(7*self.slots_per_day,cutoff-self.recent_event_horizon+1); events=[]
        for node in range(265):
            rows=np.flatnonzero(self.flags[start:cutoff+1,node])+start
            if not len(rows): continue
            starts=np.r_[0,np.flatnonzero(np.diff(rows)>1)+1]; ends=np.r_[starts[1:]-1,len(rows)-1]
            for a,b in zip(starts,ends):
                segment=rows[a:b+1]
                if len(segment)<2: continue
                ongoing=bool(segment[-1]==cutoff); peak=int(segment[np.argmax(self.scores[segment,node])])
                events.append({"node_index":node,"observed_start_row":int(segment[0]),"observed_end_row":int(segment[-1]),"state":"ongoing" if ongoing else "ended_observed","duration_observed":int(len(segment)),"peak_row_observed":peak,"peak_robust_z_observed":float(self.scores[peak,node]),"mean_robust_z_observed":float(self.scores[segment,node].mean()),"observation_cutoff":cutoff,"evidence_row_max":int(segment[-1]),"source":"feature_matrix_X.csv","derivation_method":"causal historical Q10 + robust Z; prior-day references only"})
        events=sorted(events,key=lambda e:(e["observed_end_row"],e["peak_robust_z_observed"]),reverse=True)[:self.max_events]
        return sorted(events,key=lambda e:(e["observed_start_row"],e["node_index"]))

    def build_semantic_snapshot(self,target_row_index:int)->SemanticSnapshot:
        if target_row_index<=7*self.slots_per_day or target_row_index>=len(self.traffic): raise ValueError("target row outside supported causal snapshot range")
        cutoff=target_row_index-1; events=self._events_at_cutoff(cutoff); node_ids=list(self.base_node_ids); node_types=list(self.base_node_types); features=self.static_features.copy(); dynamic=[]
        event_rows={}
        for e in events:
            node_id=f"event:grid_{e['node_index']}:start_{e['observed_start_row']}"; row=len(node_ids); event_rows[node_id]=row; node_ids.append(node_id); node_types.append("CongestionEvent")
            f=np.zeros(self.feature_dim,dtype=np.float32); f[2]=1; f[11]=min(e["duration_observed"]/12,1); f[12]=min(e["peak_robust_z_observed"]/10,1); f[13]=min(e["mean_robust_z_observed"]/10,1); f[14]=min(float(self.traffic[e["evidence_row_max"],e["node_index"]])/40,1); features=np.vstack([features,f]); dynamic.append({"node_id":node_id,**e})
        edges=[tuple(x) for x in self.static_edge_index.T]; types=self.static_edge_type.tolist()
        for e in dynamic:
            er=event_rows[e["node_id"]]; edges.append((er,e["node_index"])); types.append(self.relation_to_index["OCCURS_AT"])
            if e["peak_robust_z_observed"]>=4: edges.append((er,272)); types.append(self.relation_to_index["TRIGGERS_RULE"])
            if e["duration_observed"]>=3: edges.append((er,273)); types.append(self.relation_to_index["TRIGGERS_RULE"])
        ordered=sorted(dynamic,key=lambda e:(e["observed_start_row"],e["node_index"]))
        for a,b in zip(ordered,ordered[1:]): edges.append((event_rows[a["node_id"]],event_rows[b["node_id"]])); types.append(self.relation_to_index["TEMPORALLY_PRECEDES"])
        for a in dynamic:
            for b in dynamic:
                lag=b["observed_start_row"]-a["observed_start_row"]
                if a is not b and 0<lag<=3 and self.adjacency[a["node_index"],b["node_index"]]>0:
                    edges.append((event_rows[a["node_id"]],event_rows[b["node_id"]])); types.append(self.relation_to_index["POTENTIALLY_PROPAGATES_TO"])
        mask=np.zeros((265,len(node_ids)),dtype=bool)
        for grid in range(265):
            mask[grid,grid]=True; mask[grid,265:272]=self.poi[grid]>0; mask[grid,np.flatnonzero(self.adjacency[grid])]=True
        for e in dynamic:
            grid=e["node_index"]; er=event_rows[e["node_id"]]; mask[grid,er]=True
            if e["peak_robust_z_observed"]>=4: mask[grid,272]=True
            if e["duration_observed"]>=3: mask[grid,273]=True
        assert mask.any(1).all()
        return SemanticSnapshot(target_row_index,cutoff,node_ids,node_types,torch.tensor(features),torch.tensor(np.asarray(edges,dtype=np.int64).T),torch.tensor(types,dtype=torch.long),torch.tensor(mask),dynamic)

    def leakage_audit(self,snapshot:SemanticSnapshot)->dict[str,object]:
        max_evidence=max([d["evidence_row_max"] for d in snapshot.dynamic_metadata],default=snapshot.observation_cutoff)
        return {"target_row_index":snapshot.target_row_index,"observation_cutoff":snapshot.observation_cutoff,"maximum_dynamic_evidence_row":max_evidence,"evidence_before_target":max_evidence<snapshot.target_row_index,"ongoing_events_hide_future_end":all("final_end_row" not in d for d in snapshot.dynamic_metadata),"target_speed_in_semantic_features":False,"future_precipitation_used":False,"static_poi_treated_as_static":True}


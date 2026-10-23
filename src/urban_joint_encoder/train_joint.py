"""Device-portable Step 7A joint-pipeline smoke training entry point."""

from __future__ import annotations

import argparse, copy, json, platform, random, resource, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_joint_encoder.causal_semantic_snapshots import CausalSemanticSnapshotBuilder
from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.data_loader import SplitConfig, fit_train_standardizer, load_traffic_graph, split_boundaries
from urban_joint_encoder.joint_dataset import JointTrafficDataset, joint_collate
from urban_joint_encoder.models.joint_encoder import JointSpatiotemporalSemanticEncoder


def _json(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def resolve_device(requested:str)->torch.device:
    if requested=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def peak_memory_mb()->float:
    value=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value/(1024*1024) if platform.system()=="Darwin" else value/1024


def load_config(path:Path)->dict:
    # JSON is valid YAML; avoiding an extra parser keeps the locked environment minimal.
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)


def run_smoke_test(root:Path|None=None,config_path:Path|None=None,device_override:str|None=None):
    root=(root or find_project_root()).resolve(); config_path=config_path or root/"configs/joint_train.yaml"; cfg=load_config(config_path); set_seed(cfg["seed"]); device=resolve_device(device_override or cfg["device"])
    total_start=time.perf_counter(); traffic,adj,_=load_traffic_graph(root); bounds=split_boundaries(len(traffic),SplitConfig()); scaler=fit_train_standardizer(traffic,bounds["train"][1]); normalized=scaler.transform(traffic).astype(np.float32)
    train_rows=list(range(cfg["smoke_train_target_start"],cfg["smoke_train_target_start"]+cfg["smoke_train_window_count"])); val_rows=list(range(cfg["smoke_validation_target_start"],cfg["smoke_validation_target_start"]+cfg["smoke_validation_window_count"]))
    if not all(bounds["train"][0]<=r<bounds["train"][1] for r in train_rows): raise ValueError("Smoke train rows must remain in training split")
    if not all(bounds["validation"][0]<=r<bounds["validation"][1] for r in val_rows): raise ValueError("Smoke validation rows must remain in validation split")
    if set(train_rows)&set(val_rows): raise ValueError("Train/validation targets overlap")
    train_ds=JointTrafficDataset(normalized,train_rows,cfg["sequence_length"]); val_ds=JointTrafficDataset(normalized,val_rows,cfg["sequence_length"])
    generator=torch.Generator().manual_seed(cfg["seed"]); train_loader=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=False,generator=generator,collate_fn=joint_collate,num_workers=0); val_loader=DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,collate_fn=joint_collate,num_workers=0)
    builder=CausalSemanticSnapshotBuilder(root,cfg["recent_event_horizon"],cfg["max_dynamic_events"]); cache={}
    snapshot_times=[]
    def snapshots(rows):
        result=[]
        for row in rows:
            start=time.perf_counter()
            if row not in cache: cache[row]=builder.build_semantic_snapshot(row)
            snapshot_times.append(time.perf_counter()-start); result.append(cache[row].to(device))
        return result
    checkpoint=torch.load(root/"outputs/models/st_encoder_best.pt",map_location="cpu",weights_only=True)
    model=JointSpatiotemporalSemanticEncoder(torch.from_numpy(adj),checkpoint["model_state_dict"],cfg["hidden_dim"],6).to(device); optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg["learning_rate"]); criterion=nn.MSELoss()
    initial_attention=model.fusion.cross_attention.in_proj_weight.detach().clone(); train_history=[]; shape_log=[]; forward_seconds=0.; backward_seconds=0.; nonzero_gradient=False
    for epoch in range(1,cfg["smoke_epochs"]+1):
        model.train(); losses=[]
        for x,y,rows in train_loader:
            x=x.to(device); y=y.to(device); snaps=snapshots(rows); optimizer.zero_grad(set_to_none=True)
            tick=time.perf_counter(); output=model(x,snaps); loss=criterion(output["prediction"],y); forward_seconds+=time.perf_counter()-tick
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            tick=time.perf_counter(); loss.backward(); backward_seconds+=time.perf_counter()-tick
            grad=model.fusion.cross_attention.in_proj_weight.grad; nonzero_gradient=nonzero_gradient or bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5); optimizer.step(); losses.append(float(loss.detach().cpu()))
            if not shape_log:
                shape_log={"X_batch":list(x.shape),"target":list(y.shape),"E_st":list(output["E_st"].shape),"E_se_per_sample":[list(z.shape) for z in output["E_se"]],"attention_mask_per_sample":[list(s.attention_mask.shape) for s in snaps],"attention_weights_per_sample":[list(w.shape) for w in output["attention_weights"]],"semantic_context":list(output["semantic_context"].shape),"E_joint":list(output["E_joint"].shape),"prediction":list(output["prediction"].shape)}
            for tensor in (output["prediction"],output["E_st"],output["semantic_context"],output["E_joint"],*output["E_se"]):
                if not torch.isfinite(tensor).all(): raise RuntimeError("NaN/Inf detected in forward output")
        train_history.append({"epoch":epoch,"mean_training_mse_standardized":float(np.mean(losses))})
    parameter_updated=not torch.equal(initial_attention,model.fusion.cross_attention.in_proj_weight.detach())
    model.eval(); validation_losses=[]
    with torch.no_grad():
        for x,y,rows in val_loader:
            x=x.to(device); y=y.to(device); out=model(x,snapshots(rows)); validation_losses.append(float(criterion(out["prediction"],y).cpu()))
    # Two distinct causal snapshots must produce distinct semantic encodings.
    model.eval(); s1=cache[train_rows[0]].to(device); s2=cache[train_rows[-1]].to(device)
    with torch.no_grad(): z1=model.semantic_encoder(s1.node_features,s1.edge_index,s1.edge_type); z2=model.semantic_encoder(s2.node_features,s2.edge_index,s2.edge_type)
    shared=min(len(z1),len(z2)); semantic_snapshots_differ=bool(len(z1)!=len(z2) or not torch.allclose(z1[:shared],z2[:shared]))
    leakage_snapshots=[builder.leakage_audit(cache[row]) for row in sorted(cache)]
    leakage={"split_boundaries_half_open":{k:list(v) for k,v in bounds.items()},"train_target_rows":train_rows,"validation_target_rows":val_rows,"test_rows_used":False,"train_validation_overlap":False,"duplicate_training_windows":False,"event_threshold_reference_policy":"For row r, only prior days at the same daily slot ±1 are used; no row >= r enters its threshold.","snapshot_policy":"At target t, observation_cutoff=t-1. Ongoing events expose only observed_start through cutoff; no future final end. Recent ended events within the configured horizon are retained.","all_semantic_evidence_before_target":all(x["evidence_before_target"] for x in leakage_snapshots),"ongoing_events_do_not_expose_future_end":all(x["ongoing_events_hide_future_end"] for x in leakage_snapshots),"prediction_labels_enter_semantic_input":False,"static_poi_classification":"static only","future_precipitation_used":False,"weather_feature_policy":"Weather is omitted in Step 7A snapshots; therefore no future precipitation can enter.","step5_test_case_checkpoint_used_for_semantic_initialization":False,"semantic_encoder_initialization":"fresh random initialization with fixed seed; train-split smoke windows only","case_7347_used_in_training":7347 in train_rows,"snapshot_audits":leakage_snapshots}
    batch_count=cfg["smoke_epochs"]*len(train_loader); avg_batch=(forward_seconds+backward_seconds)/max(batch_count,1); full_batches=int(np.ceil(6036/cfg["batch_size"])); estimated_epoch=full_batches*avg_batch+6036*np.mean(snapshot_times)
    report={"stage":"Step 7A joint pipeline smoke test","component_reuse":{"reused":["data_loader header-safe loading and train-only scaler","12-step temporal window semantics","trained Step 4 SpatialTemporalEncoder weights, frozen","Step 5 relation-aware message passing design","Step 6 masked multi-head Cross-Attention with gated residual"],"extended":["causal time-indexed semantic snapshots","variable semantic-node batching","fresh train-split semantic encoder","separate joint prediction head","device-portable smoke trainer and leakage audit"]},"config":cfg,"resolved_device":str(device),"torch_version":torch.__version__,"cuda_available":torch.cuda.is_available(),"data_windows":{"full_train":6036,"full_validation":1296,"full_test":1296,"smoke_train":len(train_ds),"smoke_validation":len(val_ds)},"static_cache":{"nodes":274,"edges":int(builder.static_edge_index.shape[1]),"causal_signal_precompute_seconds":builder.signal_precompute_seconds},"snapshot_generation":{"unique_snapshots":len(cache),"total_calls":len(snapshot_times),"total_seconds":float(sum(snapshot_times)),"mean_seconds_per_call_including_cache_hits":float(np.mean(snapshot_times)),"mean_node_count":float(np.mean([len(s.node_ids) for s in cache.values()])),"variable_node_counts":sorted(set(len(s.node_ids) for s in cache.values())),"lifecycle":"ongoing and recently ended event nodes; current cutoff only; future final end hidden"},"tensor_shapes":shape_log,"training":{"history":train_history,"validation_mse_standardized":float(np.mean(validation_losses)),"cross_attention_nonzero_gradient":nonzero_gradient,"cross_attention_parameter_updated":parameter_updated,"semantic_snapshots_differ":semantic_snapshots_differ},"timing":{"forward_seconds_total":forward_seconds,"backward_seconds_total":backward_seconds,"mean_forward_backward_seconds_per_train_batch":avg_batch,"total_runtime_seconds":time.perf_counter()-total_start,"estimated_full_epoch_seconds_cpu_or_current_device":float(estimated_epoch),"estimate_basis":f"{full_batches} batches × observed smoke forward/backward plus 6036 on-demand snapshot calls; rough linear estimate"},"memory":{"peak_process_rss_mb":peak_memory_mb(),"cuda_peak_allocated_mb":float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else None},"leakage_audit_path":"outputs/metrics/joint_data_leakage_audit.json"}
    out_metrics=root/"outputs/metrics"; _json(out_metrics/"joint_pipeline_smoke_test.json",report); _json(out_metrics/"joint_data_leakage_audit.json",leakage)
    out_model=root/"outputs/models/joint_smoke_test.pt"; torch.save({"model_state_dict":model.state_dict(),"config":cfg,"note":"Smoke-test weight only; not a tuned or final model."},out_model)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/joint_train.yaml")); parser.add_argument("--device",choices=["auto","cpu","cuda"],default=None); args=parser.parse_args(); run_smoke_test(config_path=args.config,device_override=args.device)


if __name__=="__main__": main()

"""Strict metadata alignment, traceable masks, and honest single-case fusion."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.models.cross_attention import MaskedCrossModalFusion


def _load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _save_json(path,value): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def load_graph(path:Path):
    return nx.node_link_graph(_load_json(path),edges="edges")


def validate_alignment(root:Path):
    e_st=np.load(root/"outputs/embeddings/E_st.npy"); e_se=np.load(root/"outputs/embeddings/E_se.npy")
    st_meta=_load_json(root/"outputs/embeddings/E_st_metadata.json"); se_meta=_load_json(root/"outputs/embeddings/E_se_metadata.json"); graph=load_graph(root/"outputs/graphs/semantic_graph.json")
    checks={
        "E_st_shape_matches_metadata":list(e_st.shape)==st_meta["shape"]==[265,32],
        "E_se_shape_matches_metadata":list(e_se.shape)==se_meta["shape"]==[277,32],
        "E_st_case_row_is_7347":st_meta["target_row_index"]==7347,
        "E_se_case_row_is_7347":se_meta["case_row_index"]==7347,
        "same_case":st_meta["target_row_index"]==se_meta["case_row_index"],
        "E_st_node_indices_exact":st_meta["node_indices"]==list(range(265)),
        "E_se_row_mapping_bijective":sorted(se_meta["node_id_to_row_index"].values())==list(range(len(e_se))),
        "E_se_node_order_exact":[se_meta["node_id_to_row_index"][node] for node in se_meta["node_ids_in_row_order"]]==list(range(len(e_se))),
        "graph_node_set_matches_E_se":set(graph.nodes)==set(se_meta["node_ids_in_row_order"]),
    }
    graph_grids={int(d["node_index"]):n for n,d in graph.nodes(data=True) if d["type"]=="SpatialGrid"}
    checks["graph_has_exact_grid_indices"]=set(graph_grids)==set(range(265))
    checks["grid_ids_match_declared_node_index"]=all(n==f"grid:{i}" for i,n in graph_grids.items())
    checks["grid_semantic_rows_resolve"]=all(graph_grids[i] in se_meta["node_id_to_row_index"] for i in range(265))
    if not all(checks.values()): raise RuntimeError(f"Critical cross-modal alignment validation failed: {checks}")
    return e_st,e_se,st_meta,se_meta,graph,graph_grids,checks


def build_traceable_mask(graph,se_meta,graph_grids):
    node_to_row=se_meta["node_id_to_row_index"]; mask=np.zeros((265,len(node_to_row)),dtype=bool); provenance={str(i):{} for i in range(265)}
    def allow(query,node,path,category):
        row=node_to_row[node]; mask[query,row]=True; entry=provenance[str(query)].setdefault(node,{"semantic_row_index":row,"semantic_node_type":graph.nodes[node]["type"],"paths":[]}); entry["paths"].append({"category":category,"nodes":path})
    for i in range(265):
        grid=graph_grids[i]; allow(i,grid,[grid],"SELF_SPATIAL_GRID")
        for _,v,_,d in graph.out_edges(grid,keys=True,data=True):
            if d["type"]=="HAS_POI": allow(i,v,[grid,"HAS_POI",v],"HAS_POI")
            elif d["type"]=="ADJACENT_TO" and graph.nodes[v]["type"]=="SpatialGrid": allow(i,v,[grid,"ADJACENT_TO",v],"ONE_HOP_ADJACENT_GRID")
        events=[]
        for u,_,_,d in graph.in_edges(grid,keys=True,data=True):
            if d["type"]=="OCCURS_AT": events.append(u); allow(i,u,[grid,"OCCURS_AT(reverse)",u],"LOCAL_EVENT")
        for event in events:
            related={event}
            for _,v,_,d in graph.out_edges(event,keys=True,data=True):
                if d["type"] in ("TEMPORALLY_PRECEDES","POTENTIALLY_PROPAGATES_TO") and graph.nodes[v]["type"]=="CongestionEvent": related.add(v); allow(i,v,[grid,"OCCURS_AT(reverse)",event,d["type"],v],"EVENT_RELATION")
            for ev in related:
                for _,rule,_,d in graph.out_edges(ev,keys=True,data=True):
                    if d["type"]=="TRIGGERS_RULE": allow(i,rule,[grid,"OCCURS_AT(reverse)",event,"EVENT_PATH",ev,"TRIGGERS_RULE",rule],"EVENT_PATH_RULE")
    if not mask.any(1).all(): raise RuntimeError("At least one grid has no valid semantic key")
    return mask,provenance


def _layer_norm(x,eps=1e-5):
    return (x-x.mean(axis=1,keepdims=True))/np.sqrt(x.var(axis=1,keepdims=True)+eps)


def rule_fusion(e_st,e_se,provenance,se_meta,gate_value=.25):
    weights_by_category={"SELF_SPATIAL_GRID":.45,"HAS_POI":.20,"LOCAL_EVENT":.20,"ONE_HOP_ADJACENT_GRID":.10,"EVENT_RELATION":.03,"EVENT_PATH_RULE":.02}
    context=np.zeros_like(e_st,dtype=np.float32); used_weights=[]
    for i in range(len(e_st)):
        items=[]
        for node,info in provenance[str(i)].items():
            categories={p["category"] for p in info["paths"]}; w=max(weights_by_category[c] for c in categories); items.append((info["semantic_row_index"],w,node,sorted(categories)))
        total=sum(w for _,w,_,_ in items); context[i]=sum((w/total)*e_se[row] for row,w,_,_ in items); used_weights.append([{"semantic_node_id":n,"semantic_row_index":r,"normalized_weight":w/total,"categories":cats} for r,w,n,cats in items])
    st_norm=_layer_norm(e_st); context_norm=_layer_norm(context); joint=_layer_norm((1-gate_value)*st_norm+gate_value*context_norm).astype(np.float32)
    return context.astype(np.float32),joint,used_weights,weights_by_category


def run_alignment_and_fusion(root:Path|None=None):
    root=(root or find_project_root()).resolve(); e_st,e_se,st_meta,se_meta,graph,graph_grids,checks=validate_alignment(root)
    mask,provenance=build_traceable_mask(graph,se_meta,graph_grids); out=root/"outputs/alignment"; out.mkdir(parents=True,exist_ok=True); np.save(out/"attention_mask.npy",mask)
    alignment={"case_row_index":7347,"query_semantics":"265 E_st rows indexed by verified matrix node_index","key_value_semantics":"277 E_se rows resolved by E_se metadata node ID mapping","shape":list(mask.shape),"valid_element_count":int(mask.sum()),"total_element_count":int(mask.size),"coverage_ratio":float(mask.mean()),"valid_keys_per_query":{"minimum":int(mask.sum(1).min()),"maximum":int(mask.sum(1).max()),"mean":float(mask.sum(1).mean())},"validation_checks":checks,"provenance_by_query_grid":provenance}; _save_json(out/"cross_modal_alignment.json",alignment)
    # Architecture-only forward: deterministic initialization, explicitly not trained.
    torch.manual_seed(42); demo_model=MaskedCrossModalFusion(32,4); demo_model.eval()
    with torch.no_grad(): demo_context,demo_joint,attn,gate=demo_model(torch.tensor(e_st),torch.tensor(e_se),torch.tensor(mask))
    np.savez_compressed(out/"untrained_attention_forward.npz",semantic_context=demo_context.numpy(),joint=demo_joint.numpy(),attention_weights=attn.numpy(),gate=gate.numpy())
    # Honest output: deterministic traceable rule baseline, not claimed as learned fusion.
    rule_context,e_joint,rule_weights,category_weights=rule_fusion(e_st,e_se,provenance,se_meta,.25); np.save(root/"outputs/embeddings/E_joint.npy",e_joint)
    metadata={"shape":list(e_joint.shape),"dtype":str(e_joint.dtype),"case_row_index":7347,"method":"deterministic traceable rule-weighted baseline","trained":False,"fixed_semantic_gate":.25,"formula":"LayerNorm(0.75*LayerNorm(E_st) + 0.25*LayerNorm(rule_context))","rule_category_weights_before_per_query_normalization":category_weights,"query_node_indices":list(range(265)),"alignment":"outputs/alignment/cross_modal_alignment.json","warning":"This is a single-case rule fusion baseline, not a trained Cross-Attention representation and not evidence of performance improvement."}; _save_json(root/"outputs/embeddings/E_joint_metadata.json",metadata)
    audit={"alignment_validation":checks,"attention_mask":{k:alignment[k] for k in ("shape","valid_element_count","total_element_count","coverage_ratio","valid_keys_per_query")},"cross_attention_architecture":{"embedding_dim":32,"num_heads":4,"parameter_count":sum(p.numel() for p in demo_model.parameters()),"trained":False,"forward_executed":True,"attention_weights_shape":list(attn.shape),"attention_weights_source":"Actual PyTorch MultiheadAttention forward with fixed seed 42 and relation mask; weights are untrained allocation weights, not causal contributions."},"fusion_training":{"performed":False,"reason":"Only one aligned semantic Case exists; repeating it would not create independent temporal supervision.","base_encoders_frozen_and_not_retrained":True},"E_joint":{"shape":list(e_joint.shape),"method":"deterministic rule-weighted baseline","trained":False},"data_leakage":{"future_data_used_for_fusion_training":False,"fusion_training_samples":0,"single_case_repeated":False,"case_row_index":7347},"grid_146_rule_weights":rule_weights[146],"limitations":["Cross-Attention parameters are untrained; attention visualization is architectural demonstration only.","E_joint is a deterministic rule baseline, not a learned improvement.","No fusion performance claim is made from one Case.","Node identity relies on verified metadata and graph properties; geographic FIDs remain unavailable."]}; _save_json(root/"outputs/metrics/fusion_audit.json",audit)
    return audit


if __name__=="__main__": print(json.dumps(run_alignment_and_fusion(),ensure_ascii=False,indent=2))

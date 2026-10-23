"""Causal event detection, semantic graph construction, training and export for Step 5."""

from __future__ import annotations

import json, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from urban_joint_encoder.data_audit import POI_NAMES, find_project_root
from urban_joint_encoder.data_loader import load_traffic_graph, Standardizer
from urban_joint_encoder.models import STForecastModel
from urban_joint_encoder.models.semantic_encoder import RelationalGraphEncoder


RELATIONS=["HAS_POI","OCCURS_AT","ADJACENT_TO","TEMPORALLY_PRECEDES","POTENTIALLY_PROPAGATES_TO","TRIGGERS_RULE"]
NODE_TYPES=["SpatialGrid","POICategory","CongestionEvent","GovernanceRule"]


@dataclass
class DetectionConfig:
    slots_per_day:int=288
    minimum_history_days:int=7
    maximum_history_days:int=21
    slot_radius:int=1
    lower_quantile:float=.10
    robust_z_threshold:float=2.0
    minimum_speed_drop:float=2.0
    minimum_event_steps:int=2
    detection_start_row:int=7344
    detection_end_row_exclusive:int=8640
    selected_case_row:int=7347


def _json(path:Path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def load_poi(root:Path)->np.ndarray:
    base=root/"data/raw/Main data/prediction_data/POI"
    return np.vstack([pd.read_csv(base/f"poi{i}.csv",header=None,encoding="utf-8-sig").to_numpy(float).ravel() for i in range(1,8)]).T


def detect_events(traffic:np.ndarray,config:DetectionConfig):
    flags=np.zeros_like(traffic,dtype=bool); scores=np.zeros_like(traffic,dtype=np.float32); medians=np.zeros_like(traffic,dtype=np.float32); thresholds=np.zeros_like(traffic,dtype=np.float32)
    first=max(config.minimum_history_days*config.slots_per_day,config.detection_start_row)
    for row in range(first,config.detection_end_row_exclusive):
        day=row//config.slots_per_day; slot=row%config.slots_per_day
        days=np.arange(max(0,day-config.maximum_history_days),day)
        indices=np.concatenate([days*config.slots_per_day+((slot+delta)%config.slots_per_day) for delta in range(-config.slot_radius,config.slot_radius+1)])
        reference=traffic[indices]; median=np.median(reference,axis=0); q=np.quantile(reference,config.lower_quantile,axis=0); mad=np.median(np.abs(reference-median),axis=0)
        score=(median-traffic[row])/(1.4826*mad+.25); flag=(traffic[row]<=q)&(score>=config.robust_z_threshold)&((median-traffic[row])>=config.minimum_speed_drop)
        flags[row]=flag; scores[row]=score; medians[row]=median; thresholds[row]=q
    events=[]
    for node in range(traffic.shape[1]):
        rows=np.flatnonzero(flags[config.detection_start_row:config.detection_end_row_exclusive,node])+config.detection_start_row
        if not len(rows): continue
        starts=np.r_[0,np.flatnonzero(np.diff(rows)>1)+1]; ends=np.r_[starts[1:]-1,len(rows)-1]
        for a,b in zip(starts,ends):
            segment=rows[a:b+1]
            if len(segment)<config.minimum_event_steps: continue
            peak=int(segment[np.argmax(scores[segment,node])])
            events.append({"event_id":f"event:grid_{node}:rows_{int(segment[0])}_{int(segment[-1])}","node_index":node,
                "start_row_index":int(segment[0]),"end_row_index":int(segment[-1]),"duration_steps":int(len(segment)),"peak_row_index":peak,
                "peak_robust_z":float(scores[peak,node]),"mean_robust_z":float(scores[segment,node].mean()),"mean_observed_speed":float(traffic[segment,node].mean()),
                "peak_observed_speed":float(traffic[peak,node]),"peak_historical_median":float(medians[peak,node]),"peak_historical_q10":float(thresholds[peak,node]),
                "evidence":f"Observed speed stayed below causal historical Q10 with robust Z >= {config.robust_z_threshold} for {len(segment)} consecutive steps.",
                "generation_method":"derived","interpretation":"Traffic-speed anomaly consistent with congestion; not evidence of an accident or cause."})
    return events,flags,scores


def schema_definition():
    return {"node_types":{
        "SpatialGrid":{"unique_id":"grid:{matrix_column_index}","properties":["node_index","poi_counts"],"source":"headerless traffic/POI matrices","generation_method":"observed"},
        "POICategory":{"unique_id":"poi_category:{index}","properties":["category_name"],"source":"author POI query categories","generation_method":"observed"},
        "CongestionEvent":{"unique_id":"event:grid_{node}:rows_{start}_{end}","properties":["row range","robust anomaly evidence"],"source":"traffic speed matrix","generation_method":"derived"},
        "GovernanceRule":{"unique_id":"rule:{name}","properties":["condition","suggested_action"],"source":"project demonstration only","generation_method":"illustrative"}},
      "relationship_types":{
        "HAS_POI":{"source":"POI count files","generation_method":"observed","evidence":"positive per-grid category count"},
        "OCCURS_AT":{"source":"event detector","generation_method":"derived","evidence":"event node_index"},
        "ADJACENT_TO":{"source":"adj_matrix.csv","generation_method":"observed","evidence":"binary matrix entry"},
        "TEMPORALLY_PRECEDES":{"source":"event row ranges","generation_method":"derived","evidence":"ordered non-identical starts"},
        "POTENTIALLY_PROPAGATES_TO":{"source":"adjacency + event timing + anomaly strength","generation_method":"inferred","evidence":"candidate association; explicitly non-causal"},
        "TRIGGERS_RULE":{"source":"demonstration threshold","generation_method":"illustrative","evidence":"rule condition evaluated on derived event"}}}


def build_graph(traffic,adjacency,poi,events,config):
    graph=nx.MultiDiGraph(schema_version="1.0",case_row_index=config.selected_case_row,layout_semantics="topological_not_geographic")
    for node in range(traffic.shape[1]): graph.add_node(f"grid:{node}",id=f"grid:{node}",type="SpatialGrid",node_index=node,poi_counts=json.dumps(poi[node].astype(int).tolist()),data_source="POI CSV + matrix column",evidence="matrix column index",generation_method="observed")
    for i,name in enumerate(POI_NAMES): graph.add_node(f"poi_category:{i}",id=f"poi_category:{i}",type="POICategory",category_index=i,category_name=name,data_source="POI_class_num.ipynb",evidence="source query whitelist and sorted export order",generation_method="observed")
    for node in range(len(poi)):
        for i,count in enumerate(poi[node]):
            if count>0: graph.add_edge(f"grid:{node}",f"poi_category:{i}",key=f"has_poi:{node}:{i}",id=f"has_poi:{node}:{i}",type="HAS_POI",count=int(count),data_source=f"poi{i+1}.csv",evidence=f"observed count={int(count)}",generation_method="observed")
    for i,j in zip(*np.where(np.triu(adjacency,1)>0)):
        for src,dst in ((i,j),(j,i)): graph.add_edge(f"grid:{src}",f"grid:{dst}",key=f"adj:{src}:{dst}",id=f"adj:{src}:{dst}",type="ADJACENT_TO",data_source="adj_matrix.csv",evidence="binary adjacency entry=1",generation_method="observed")
    # Case-local event subgraph: all events overlapping the selected row, plus the selected segment.
    selected=next(e for e in events if e["node_index"]==146 and e["start_row_index"]==7345 and e["end_row_index"]==7349)
    case_events=[e for e in events if e["start_row_index"]<=config.selected_case_row<=e["end_row_index"]]
    if selected not in case_events: case_events.append(selected)
    for e in case_events:
        attrs={k:v for k,v in e.items() if k!="event_id"}; graph.add_node(e["event_id"],id=e["event_id"],type="CongestionEvent",data_source="feature_matrix_X.csv",**attrs)
        graph.add_edge(e["event_id"],f"grid:{e['node_index']}",key=f"occurs:{e['event_id']}",id=f"occurs:{e['event_id']}",type="OCCURS_AT",data_source="causal seasonal anomaly detector",evidence=f"node_index={e['node_index']}",generation_method="derived")
    ordered=sorted(case_events,key=lambda e:(e["start_row_index"],e["node_index"]))
    for a,b in zip(ordered,ordered[1:]):
        graph.add_edge(a["event_id"],b["event_id"],key=f"precedes:{a['event_id']}:{b['event_id']}",id=f"precedes:{a['event_id']}:{b['event_id']}",type="TEMPORALLY_PRECEDES",data_source="derived event row ranges",evidence=f"start {a['start_row_index']} <= {b['start_row_index']}",generation_method="derived")
    for a in case_events:
        for b in case_events:
            lag=b["start_row_index"]-a["start_row_index"]
            if a is not b and adjacency[a["node_index"],b["node_index"]]>0 and 0<lag<=3:
                confidence=min(1.0,.45+.05*a["peak_robust_z"]+.05*b["peak_robust_z"]-.05*lag)
                evidence=f"adjacent grids; downstream event starts {lag} steps later; source/destination peak robust Z={a['peak_robust_z']:.2f}/{b['peak_robust_z']:.2f}"
                graph.add_edge(a["event_id"],b["event_id"],key=f"candidate:{a['event_id']}:{b['event_id']}",id=f"candidate:{a['event_id']}:{b['event_id']}",type="POTENTIALLY_PROPAGATES_TO",lag_steps=lag,confidence=float(confidence),causal_claim=False,data_source="adjacency + event detector",evidence=evidence,generation_method="inferred")
    rules=[("rule:monitor_adjacent","Monitor adjacent grid anomalies","If derived event peak robust Z >= 4, inspect adjacent-grid indicators; demonstration only."),("rule:review_poi_context","Review local POI context","If event lasts >= 3 steps, review observed POI profile; demonstration only.")]
    for rid,label,condition in rules: graph.add_node(rid,id=rid,type="GovernanceRule",label=label,condition=condition,suggested_action=label,data_source="project demonstration",evidence="no official policy or action record",generation_method="illustrative")
    for e in case_events:
        targets=[]
        if e["peak_robust_z"]>=4: targets.append("rule:monitor_adjacent")
        if e["duration_steps"]>=3: targets.append("rule:review_poi_context")
        for rid in targets: graph.add_edge(e["event_id"],rid,key=f"triggers:{e['event_id']}:{rid}",id=f"triggers:{e['event_id']}:{rid}",type="TRIGGERS_RULE",data_source="illustrative rule evaluation",evidence=graph.nodes[rid]["condition"],generation_method="illustrative")
    return graph,selected,case_events


def graph_tensors(graph,poi):
    node_ids=list(graph.nodes); node_to_index={node:i for i,node in enumerate(node_ids)}; rel_to_index={r:i for i,r in enumerate(RELATIONS)}
    features=np.zeros((len(node_ids),18),dtype=np.float32)
    for i,node in enumerate(node_ids):
        a=graph.nodes[node]; features[i,NODE_TYPES.index(a["type"])]=1
        if a["type"]=="SpatialGrid": features[i,4:11]=np.log1p(poi[a["node_index"]])/np.log1p(np.maximum(poi.max(0),1))
        elif a["type"]=="POICategory": features[i,11+a["category_index"]]=1
        elif a["type"]=="CongestionEvent": features[i,11]=min(a["duration_steps"]/12,1); features[i,12]=min(a["peak_robust_z"]/10,1); features[i,13]=min(a["mean_robust_z"]/10,1); features[i,14]=min(a["mean_observed_speed"]/40,1)
        elif a["type"]=="GovernanceRule": features[i,15]=1; features[i,16]=float("adjacent" in a["condition"].lower()); features[i,17]=float("poi" in a["condition"].lower())
    edges=list(graph.edges(keys=True,data=True)); edge_index=np.array([[node_to_index[u],node_to_index[v]] for u,v,_,_ in edges],dtype=np.int64).T; edge_type=np.array([rel_to_index[d["type"]] for *_,d in edges],dtype=np.int64)
    return node_ids,node_to_index,rel_to_index,features,edge_index,edge_type


def train_encoder(root,graph,poi,seed=42,epochs=160,hidden_dim=32):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    node_ids,node_to_index,rel_to_index,features,edge_index,edge_type=graph_tensors(graph,poi); rng=np.random.default_rng(seed)
    order=rng.permutation(edge_index.shape[1]); split=max(1,int(.8*len(order))); train_idx=order[:split]; val_idx=order[split:]
    x=torch.tensor(features); train_edges=torch.tensor(edge_index[:,train_idx]); train_types=torch.tensor(edge_type[train_idx]); val_edges=torch.tensor(edge_index[:,val_idx]); val_types=torch.tensor(edge_type[val_idx])
    existing={(int(edge_index[0,i]),int(edge_index[1,i]),int(edge_type[i])) for i in range(edge_index.shape[1])}
    def negatives(count,types):
        result=[]
        while len(result)<count:
            s=int(rng.integers(len(node_ids))); d=int(rng.integers(len(node_ids))); r=int(types[len(result)%len(types)])
            if s!=d and (s,d,r) not in existing: result.append((s,d,r))
        a=np.array(result); return torch.tensor(a[:,:2].T),torch.tensor(a[:,2])
    train_neg,train_neg_types=negatives(len(train_idx),edge_type[train_idx]); val_neg,val_neg_types=negatives(len(val_idx),edge_type[val_idx])
    model=RelationalGraphEncoder(features.shape[1],hidden_dim,len(RELATIONS)); optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-4); bce=torch.nn.BCEWithLogitsLoss(); mse=torch.nn.MSELoss()
    best=-1.; best_epoch=0; stale=0; history=[]; path=root/"outputs/models/semantic_encoder_best.pt"; path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(); z=model(x,train_edges,train_types)
        pos=model.edge_logits(z,train_edges,train_types); neg=model.edge_logits(z,train_neg,train_neg_types)
        edge_loss=bce(torch.cat([pos,neg]),torch.cat([torch.ones_like(pos),torch.zeros_like(neg)])); attr_loss=mse(model.reconstruct_attributes(z),x); loss=edge_loss+.25*attr_loss; loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            zv=model(x,train_edges,train_types); vp=model.edge_logits(zv,val_edges,val_types); vn=model.edge_logits(zv,val_neg,val_neg_types); labels=np.r_[np.ones(len(vp)),np.zeros(len(vn))]; probs=torch.sigmoid(torch.cat([vp,vn])).numpy(); auc=float(roc_auc_score(labels,probs)); vloss=float(bce(torch.cat([vp,vn]),torch.tensor(labels,dtype=torch.float32)))
        history.append({"epoch":epoch,"training_loss":float(loss.detach()),"edge_reconstruction_loss":float(edge_loss.detach()),"attribute_reconstruction_loss":float(attr_loss.detach()),"validation_edge_bce":vloss,"validation_edge_auc":auc})
        if auc>best+1e-4: best=auc; best_epoch=epoch; stale=0; torch.save({"model_state_dict":model.state_dict(),"input_dim":features.shape[1],"hidden_dim":hidden_dim,"relation_types":RELATIONS,"node_types":NODE_TYPES},path)
        else:
            stale+=1
            if stale>=25: break
    ckpt=torch.load(path,map_location="cpu",weights_only=True); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad(): embedding=model(x,train_edges,train_types).numpy().astype(np.float32)
    return embedding,{"objective":"relation-aware edge reconstruction (BCE) + node attribute reconstruction (0.25 × MSE)","seed":seed,"epochs_completed":len(history),"best_epoch":best_epoch,"best_validation_edge_auc":best,"history":history,"train_edge_count":len(train_idx),"validation_edge_count":len(val_idx),"negative_sampling":"equal number of fixed-seed corrupt non-edges per split","parameter_count":sum(p.numel() for p in model.parameters())},node_ids,node_to_index,rel_to_index


def export_aligned_est(root,row_index):
    traffic,adj,_=load_traffic_graph(root); ckpt=torch.load(root/"outputs/models/st_encoder_best.pt",map_location="cpu",weights_only=True); sc=ckpt["scaler"]; scaler=Standardizer(**sc)
    x=scaler.transform(traffic[row_index-12:row_index]).astype(np.float32); model=STForecastModel(torch.from_numpy(adj),32); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad(): _,z=model(torch.tensor(x).unsqueeze(0).unsqueeze(-1))
    np.save(root/"outputs/embeddings/E_st.npy",z[0].numpy().astype(np.float32)); meta={"shape":[265,32],"dtype":"float32","target_row_index":row_index,"history_row_range_inclusive":[row_index-12,row_index-1],"node_indices":list(range(265)),"node_index_semantics":"Matrix column index only; not a geographic FID and not a coordinate.","timestamp_status":"Unavailable in source export; no timestamp fabricated.","checkpoint":"outputs/models/st_encoder_best.pt","standardizer":scaler.as_dict(),"reexport_reason":"Aligned to Step 5 selected anomaly case."}; _json(root/"outputs/embeddings/E_st_metadata.json",meta)


def run_semantic_pipeline(root:Path|None=None):
    root=(root or find_project_root()).resolve(); started=time.perf_counter(); config=DetectionConfig(); traffic,adj,_=load_traffic_graph(root); poi=load_poi(root)
    events,flags,scores=detect_events(traffic,config); graph,selected,case_events=build_graph(traffic,adj,poi,events,config); export_aligned_est(root,config.selected_case_row)
    graphs=root/"outputs/graphs"; graphs.mkdir(parents=True,exist_ok=True)
    data=nx.node_link_data(graph,edges="edges"); data["schema"]=schema_definition(); _json(graphs/"semantic_graph.json",data); nx.write_graphml(graph,graphs/"semantic_graph.graphml")
    event_report={"config":asdict(config),"method":"Causal seasonal historical Q10 plus robust Z-score using only earlier days at same slot ±1; MAD denominator stabilized by 0.25.","detected_event_count":len(events),"events":events,"selected_case":{"case_row_index":config.selected_case_row,"selected_event_id":selected["event_id"],"selected_event":selected,"case_event_ids":[e["event_id"] for e in case_events]},"terminology":"All events are derived speed anomalies consistent with congestion; none is labeled as an accident."}; _json(root/"outputs/metrics/event_detection.json",event_report)
    embedding,training,node_ids,node_to_index,rel_to_index=train_encoder(root,graph,poi); np.save(root/"outputs/embeddings/E_se.npy",embedding)
    metadata={"shape":list(embedding.shape),"dtype":str(embedding.dtype),"semantic_hidden_dim":32,"node_ids_in_row_order":node_ids,"node_id_to_row_index":node_to_index,"node_type_by_row":[graph.nodes[n]["type"] for n in node_ids],"relation_type_to_index":rel_to_index,"case_row_index":config.selected_case_row,"aligned_E_st_target_row_index":config.selected_case_row,"scope":"All typed semantic nodes; not forced to [265,32].","checkpoint":"outputs/models/semantic_encoder_best.pt"}; _json(root/"outputs/embeddings/E_se_metadata.json",metadata)
    node_counts={t:sum(d["type"]==t for _,d in graph.nodes(data=True)) for t in NODE_TYPES}; edge_counts={r:sum(d["type"]==r for *_,d in graph.edges(data=True)) for r in RELATIONS}
    audit={"schema":schema_definition(),"graph":{"nodes":graph.number_of_nodes(),"edges":graph.number_of_edges(),"node_counts_by_type":node_counts,"edge_counts_by_type":edge_counts,"is_multidigraph":True},"selected_case":event_report["selected_case"],"semantic_encoder_training":training,"embedding":{"path":"outputs/embeddings/E_se.npy","shape":list(embedding.shape)},"alignment":{"E_st_target_row_index":config.selected_case_row,"E_se_case_row_index":config.selected_case_row,"same_case":True},"limitations":["No timestamps, geographic FIDs or coordinates are available.","Speed anomaly is not an accident label and does not identify cause.","POTENTIALLY_PROPAGATES_TO is a candidate association, not causal evidence.","GovernanceRule nodes and TRIGGERS_RULE edges are illustrative, not official actions or policy.","Graph encoder learns from this single constructed case graph; external generalization is unverified."],"runtime_seconds":time.perf_counter()-started}; _json(root/"outputs/metrics/semantic_graph_audit.json",audit)
    return audit


if __name__=="__main__": print(json.dumps(run_semantic_pipeline(),ensure_ascii=False,indent=2))

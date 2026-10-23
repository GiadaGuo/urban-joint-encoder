"""PPT-ready figures for the Step 5 semantic event graph."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

from urban_joint_encoder.data_loader import load_traffic_graph
from urban_joint_encoder.semantic_pipeline import NODE_TYPES, RELATIONS


TYPE_COLORS={"SpatialGrid":"#56CFE1","POICategory":"#FFB000","CongestionEvent":"#F28482","GovernanceRule":"#B983FF"}
METHOD_COLORS={"observed":"#56CFE1","derived":"#FFB000","inferred":"#F28482","illustrative":"#B983FF"}


def _style():
    plt.rcParams.update({"figure.facecolor":"#08111F","axes.facecolor":"#0D1B2A","axes.edgecolor":"#7693B3","axes.labelcolor":"#E8F1FA","text.color":"#E8F1FA","xtick.color":"#C5D5E5","ytick.color":"#C5D5E5","grid.color":"#38516B","font.size":12,"axes.titleweight":"bold","axes.titlesize":19,"font.family":"sans-serif","font.sans-serif":["PingFang SC","Hiragino Sans GB","Arial Unicode MS","DejaVu Sans"],"axes.unicode_minus":False})


def _save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True); png=out/f"{stem}.png"; svg=out/f"{stem}.svg"; fig.savefig(png,dpi=300,facecolor=fig.get_facecolor()); fig.savefig(svg,facecolor=fig.get_facecolor()); plt.show(); plt.close(fig); return [str(png),str(svg)]


def schema_figure(out):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    boxes={"SpatialGrid":(1,5.8),"POICategory":(1,1.5),"CongestionEvent":(11.5,5.8),"GovernanceRule":(11.5,1.5)}
    for typ,(x,y) in boxes.items(): ax.add_patch(FancyBboxPatch((x,y),3.3,1.45,boxstyle="round,pad=.08",fc=TYPE_COLORS[typ],ec="white",alpha=.85)); ax.text(x+1.65,y+.72,typ,ha="center",va="center",weight="bold",fontsize=16)
    relations=[("SpatialGrid","POICategory","HAS_POI","observed"),("CongestionEvent","SpatialGrid","OCCURS_AT","derived"),("SpatialGrid","SpatialGrid","ADJACENT_TO","observed"),("CongestionEvent","CongestionEvent","TEMPORALLY_PRECEDES","derived"),("CongestionEvent","CongestionEvent","POTENTIALLY_PROPAGATES_TO","inferred"),("CongestionEvent","GovernanceRule","TRIGGERS_RULE","illustrative")]
    ys=[7.8,7.15,6.5,5.15,4.5,3.85]
    for (s,t,r,m),y in zip(relations,ys): ax.text(8,y,f"{r}  •  {m}",ha="center",color=METHOD_COLORS[m],fontsize=13,weight="bold")
    ax.set_title("Urban Event Semantic Graph Schema • Provenance-aware Types",pad=18)
    ax.text(8,.5,"Observed = source matrices • Derived = algorithm output • Inferred = non-causal candidate • Illustrative = no official authority",ha="center",color="#C5D5E5")
    return _save(fig,out,"figure_11_semantic_graph_schema")


def timeline(root,report,out):
    _style(); traffic,_,_=load_traffic_graph(root); e=report["selected_case"]["selected_event"]; node=e["node_index"]; start,end=e["start_row_index"]-24,e["end_row_index"]+24; rows=np.arange(start,end+1); med=[]; q10=[]; S=288
    for row in rows:
        day=row//S; slot=row%S; days=np.arange(max(0,day-21),day); idx=np.concatenate([days*S+((slot+d)%S) for d in(-1,0,1)]); ref=traffic[idx,node]; med.append(np.median(ref)); q10.append(np.quantile(ref,.1))
    fig,ax=plt.subplots(figsize=(16,9)); ax.plot(rows,traffic[rows,node],color="#F7F7FF",lw=2,label="Observed speed"); ax.plot(rows,med,color="#56CFE1",lw=2,label="Causal historical median"); ax.plot(rows,q10,color="#FFB000",lw=2,ls="--",label="Causal historical Q10")
    ax.axvspan(e["start_row_index"],e["end_row_index"],color="#F28482",alpha=.28,label="Derived anomaly segment"); ax.axvline(report["selected_case"]["case_row_index"],color="#B983FF",ls=":",lw=2.5,label="Unified case row 7347")
    ax.set(title=f"Real Speed Anomaly Timeline • Matrix Node {node}",xlabel="Original row_index (timestamps unavailable)",ylabel="Traffic speed (unit pending confirmation)"); ax.grid(alpha=.22); ax.legend(frameon=False,ncol=3)
    ax.text(.01,.03,f"Rows {e['start_row_index']}–{e['end_row_index']} • peak robust Z={e['peak_robust_z']:.2f} • derived congestion-like anomaly, not an accident label",transform=ax.transAxes,color="#F4B6B6")
    return _save(fig,out,"figure_12_event_detection_timeline")


def evidence_chain(graph,report,out):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off"); case=report["selected_case"]; ids=case["case_event_ids"]
    items=[(1,5.3,"Observed speeds\nfeature matrix","observed"),(4.2,5.3,"Event @ grid 146\nrows 7345–7349\npeak Z 5.88","derived"),(7.5,5.3,"Adjacent grid 147\nevent rows 7347–7348","derived"),(10.8,5.3,"Candidate propagation\nlag=2 steps\nNON-CAUSAL","inferred"),(13.2,2,"Demo rule\nmonitor indicators","illustrative")]
    for x,y,label,m in items: ax.add_patch(FancyBboxPatch((x,y),2.2,1.55,boxstyle="round,pad=.08",fc=METHOD_COLORS[m],ec="white",alpha=.83)); ax.text(x+1.1,y+.78,label,ha="center",va="center",fontsize=12,weight="bold")
    for a,b in [((3.2,6.05),(4.2,6.05)),((6.4,6.05),(7.5,6.05)),((9.7,6.05),(10.8,6.05)),((11.9,5.3),(14.1,3.55))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=18,color="#E8F1FA",lw=2))
    ax.text(8,8,"Event Relation Chain and Auditable Evidence",ha="center",fontsize=23,weight="bold")
    ax.text(8,.65,"Evidence: causal historical threshold + robust Z • adjacency matrix entry 146 <-> 147 • temporal lag • no accident or official-action claim",ha="center",color="#C5D5E5")
    return _save(fig,out,"figure_13_event_relation_evidence")


def knowledge_graph(graph,out):
    _style(); event_nodes=[n for n,d in graph.nodes(data=True) if d["type"]=="CongestionEvent"]; focus=set(event_nodes)
    for e in event_nodes:
        focus.update(v for _,v,_,_ in graph.out_edges(e,keys=True,data=True)); focus.update(u for u,_,_,_ in graph.in_edges(e,keys=True,data=True))
    grid_nodes=[n for n in focus if graph.nodes[n]["type"]=="SpatialGrid"]
    for g in grid_nodes: focus.update(v for _,v,_,d in graph.out_edges(g,keys=True,data=True) if d["type"]=="HAS_POI")
    sub=graph.subgraph(focus).copy(); raw_pos=nx.spring_layout(nx.Graph(sub),seed=42,k=1.15,iterations=180)
    # Reserve a dedicated left sidebar for the legend so it cannot cover graph content.
    pos={node:(.29+.66*(xy[0]+1)/2,.10+.80*(xy[1]+1)/2) for node,xy in raw_pos.items()}
    fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); nx.draw_networkx_edges(sub,pos,ax=ax,edge_color="#7088A0",alpha=.45,width=1.1,arrows=True,arrowsize=12)
    for typ,color in TYPE_COLORS.items():
        nodes=[n for n in sub if sub.nodes[n]["type"]==typ]; nx.draw_networkx_nodes(sub,pos,nodelist=nodes,node_color=color,node_size=650 if typ=="CongestionEvent" else 360,edgecolors="white",linewidths=.8,label=typ,ax=ax)
    labels={n:(f"grid {sub.nodes[n]['node_index']}" if sub.nodes[n]["type"]=="SpatialGrid" else sub.nodes[n].get("category_name",sub.nodes[n].get("label",n.split(':')[0]))) for n in sub}; nx.draw_networkx_labels(sub,pos,labels,font_size=8,font_color="white",ax=ax)
    ax.set_title("Case-local Multi-type Urban Semantic Knowledge Graph",pad=18)
    legend=ax.legend(
        title="Node type", frameon=True, ncol=1, loc="upper left",
        bbox_to_anchor=(.015,.96), borderpad=.8, labelspacing=.7,
        facecolor="#0D1B2A", edgecolor="#7693B3", framealpha=.92,
    )
    legend.get_title().set_fontweight("bold")
    ax.text(.5,.018,"Topology layout only — not geographic • full graph contains all 265 grids and 2,050 typed edges",transform=ax.transAxes,ha="center",color="#56CFE1")
    return _save(fig,out,"figure_14_multitype_semantic_graph")


def encoder_process(audit,out):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    items=[(0.5,"Typed attributes\n[277,18]"),(3.6,"Relation-specific\nmessage weights ×6"),(7,"Aggregate + self\n2 R-GCN layers"),(10.4,"E_se\n[277,32]"),(13.3,"Edge + attribute\nreconstruction")]
    for i,(x,label) in enumerate(items): ax.add_patch(FancyBboxPatch((x,3.45),2.2,1.8,boxstyle="round,pad=.08",fc=list(TYPE_COLORS.values())[i%4],ec="white",alpha=.82)); ax.text(x+1.1,4.35,label,ha="center",va="center",weight="bold",fontsize=14)
    for x in [2.7,5.8,9.2,12.6]: ax.add_patch(FancyArrowPatch((x,4.35),(x+.8,4.35),arrowstyle="-|>",mutation_scale=18,color="white",lw=2))
    tr=audit["semantic_encoder_training"]; ax.set_title("Relation-aware Semantic Encoder • Learned Message Propagation",pad=18)
    ax.text(8,7.2,"Messages are transformed by HAS_POI / OCCURS_AT / ADJACENT_TO / temporal / candidate / rule relation type",ha="center",color="#9BDFF0")
    ax.text(8,1.4,f"Training objective: edge BCE + 0.25 × attribute MSE • best validation edge AUC {tr['best_validation_edge_auc']:.4f} • best epoch {tr['best_epoch']}",ha="center",fontsize=13)
    return _save(fig,out,"figure_15_semantic_encoder_messages")


def embedding_plot(embedding,metadata,out):
    _style(); pca=PCA(2); xy=pca.fit_transform(embedding); types=np.array(metadata["node_type_by_row"]); fig,ax=plt.subplots(figsize=(16,9))
    for typ,color in TYPE_COLORS.items():
        mask=types==typ; ax.scatter(xy[mask,0],xy[mask,1],s=46 if typ=="SpatialGrid" else 100,c=color,label=f"{typ} ({mask.sum()})",alpha=.85,edgecolors="white",linewidths=.35)
    for i,t in enumerate(types):
        if t in ("CongestionEvent","GovernanceRule"): ax.annotate(metadata["node_ids_in_row_order"][i].replace("event:","evt ").replace("rule:","rule "),xy[i],xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set(title="Learned E_se Semantic Node Embeddings • PCA",xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=2)
    ax.text(.01,.02,"Embeddings are trained by relation and attribute reconstruction; no random untrained vectors are presented",transform=ax.transAxes,color="#56CFE1")
    return _save(fig,out,"figure_16_ese_embedding_pca")


def load_graph(path):
    data=json.loads(path.read_text(encoding="utf-8")); return nx.node_link_graph(data,edges="edges")


def make_step5_figures(root:Path):
    out=root/"outputs/figures"; report=json.loads((root/"outputs/metrics/event_detection.json").read_text()); audit=json.loads((root/"outputs/metrics/semantic_graph_audit.json").read_text()); metadata=json.loads((root/"outputs/embeddings/E_se_metadata.json").read_text()); embedding=np.load(root/"outputs/embeddings/E_se.npy"); graph=load_graph(root/"outputs/graphs/semantic_graph.json")
    return {"figure_11":schema_figure(out),"figure_12":timeline(root,report,out),"figure_13":evidence_chain(graph,report,out),"figure_14":knowledge_graph(graph,out),"figure_15":encoder_process(audit,out),"figure_16":embedding_plot(embedding,metadata,out)}

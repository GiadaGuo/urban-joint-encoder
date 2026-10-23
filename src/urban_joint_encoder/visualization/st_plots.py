"""PPT-ready Step 4 visualizations based on trained artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from urban_joint_encoder.data_loader import load_traffic_graph


COLORS=["#56CFE1","#FFB000","#B983FF","#F28482","#90BE6D"]


def _style():
    plt.rcParams.update({"figure.facecolor":"#08111F","axes.facecolor":"#0D1B2A","axes.edgecolor":"#7693B3",
        "axes.labelcolor":"#E8F1FA","text.color":"#E8F1FA","xtick.color":"#C5D5E5","ytick.color":"#C5D5E5",
        "grid.color":"#38516B","font.size":12,"axes.titleweight":"bold","axes.titlesize":19,
        "font.family":"sans-serif","font.sans-serif":["PingFang SC","Hiragino Sans GB","Arial Unicode MS","DejaVu Sans"],"axes.unicode_minus":False})


def _save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True); png=out/f"{stem}.png"; svg=out/f"{stem}.svg"
    fig.savefig(png,dpi=300,facecolor=fig.get_facecolor()); fig.savefig(svg,facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return [str(png),str(svg)]


def architecture(out: Path):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    boxes=[(0.5,3.25,2.3,2,"Input windows\n[B,12,265,1]","#16324F"),(3.5,3.25,2.5,2,"Symmetric graph\nnormalization\n[265,265]","#184E77"),
           (6.7,3.25,2.4,2,"GCN aggregation\n+ Linear + ReLU\n[B,12,265,32]","#1A759F"),(9.8,3.25,2.3,2,"Node-wise GRU\nthrough 12 steps\n[B,265,32]","#5A189A"),
           (12.8,5.4,2.5,1.5,"Encoder output E_st\n[B,265,32]","#7B2CBF"),(12.8,1.3,2.5,1.5,"Linear prediction head\n[B,265]","#A44A3F")]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08,rounding_size=.15",fc=color,ec="#E8F1FA",lw=1.4)); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=14,weight="bold")
    for a,b in [((2.8,4.25),(3.5,4.25)),((6,4.25),(6.7,4.25)),((9.1,4.25),(9.8,4.25)),((12.1,4.25),(12.8,6.15)),((12.1,4.25),(12.8,2.05))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=18,color="#56CFE1",lw=2))
    ax.text(8,8.05,"Spatiotemporal Graph Encoder • Tensor Flow",ha="center",fontsize=24,weight="bold")
    ax.text(8,.45,"Encoder and prediction head are explicitly separated • learned on real 5-minute traffic sequences",ha="center",color="#9BDFF0")
    return _save(fig,out,"figure_05_stgnn_architecture")


def propagation(adjacency: np.ndarray,out: Path):
    _style(); graph=nx.from_numpy_array(adjacency); center=max(graph.degree,key=lambda x:x[1])[0]
    nodes=set([center]); nodes.update(graph.neighbors(center));
    for n in list(nodes): nodes.update(graph.neighbors(n))
    sub=graph.subgraph(sorted(nodes)).copy(); pos=nx.spring_layout(sub,seed=42,k=.65,iterations=150)
    distance=nx.single_source_shortest_path_length(sub,center,cutoff=2); color=[distance.get(n,2) for n in sub.nodes]
    fig,ax=plt.subplots(figsize=(16,9)); ax.axis("off")
    nx.draw_networkx_edges(sub,pos,ax=ax,edge_color="#56738F",alpha=.55,width=1.2)
    nx.draw_networkx_nodes(sub,pos,ax=ax,node_color=color,cmap="plasma_r",node_size=[520 if n==center else 220 for n in sub.nodes],edgecolors="#F7F7FF",linewidths=1)
    nx.draw_networkx_labels(sub,pos,ax=ax,font_color="white",font_size=9)
    ax.set_title(f"GCN Neighborhood Propagation • Center Node {center} (degree {graph.degree[center]})",pad=18)
    ax.text(.5,.02,"Actual adjacency • self-loops added during normalization • spring topology layout, not a geographic map",transform=ax.transAxes,ha="center",color="#56CFE1")
    return _save(fig,out,"figure_06_gcn_neighborhood_propagation")


def hidden_states(hidden: np.ndarray,metadata: dict,out: Path):
    _style(); norms=np.linalg.norm(hidden,axis=2); variability=norms.std(axis=0); chosen=np.argsort(variability)[-4:][::-1]
    fig,ax=plt.subplots(figsize=(16,9)); x=np.arange(hidden.shape[0])
    for n,c in zip(chosen,COLORS): ax.plot(x,norms[:,n],marker="o",lw=2.2,color=c,label=f"Node {n}")
    ax.set(title=f"Temporal Hidden-state Evolution • Target Row {metadata['target_row_index']}",xlabel="History step within the real 12-step window",ylabel="GRU hidden-state L2 norm")
    ax.set_xticks(x); ax.grid(alpha=.25); ax.legend(frameon=False,ncol=4)
    ax.text(.01,.03,f"History rows {metadata['history_row_range_inclusive'][0]}–{metadata['history_row_range_inclusive'][1]} • nodes selected by actual hidden-state variability",transform=ax.transAxes,color="#56CFE1")
    return _save(fig,out,"figure_07_temporal_hidden_states")


def embedding_pca(embedding: np.ndarray,out: Path):
    _style(); projection=PCA(n_components=2).fit_transform(embedding); pca=PCA(n_components=2).fit(embedding); magnitude=np.linalg.norm(embedding,axis=1)
    fig,ax=plt.subplots(figsize=(16,9)); scatter=ax.scatter(projection[:,0],projection[:,1],c=magnitude,cmap="viridis",s=55,alpha=.88,edgecolors="#E8F1FA",linewidths=.35)
    for n in np.argsort(magnitude)[-5:]: ax.annotate(str(n),projection[n],xytext=(5,5),textcoords="offset points",fontsize=9)
    cb=fig.colorbar(scatter,ax=ax,pad=.015); cb.set_label("Embedding L2 norm")
    ax.set(title="Learned E_st Node Embeddings • PCA Projection",xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.grid(alpha=.2); ax.text(.01,.02,"265 learned node states from one real test window • node labels are matrix column indices",transform=ax.transAxes,color="#56CFE1")
    return _save(fig,out,"figure_08_est_embedding_pca")


def prediction_plot(arrays,out: Path):
    _style(); pred=arrays["predictions"]; true=arrays["targets"]; rows=arrays["target_row_indices"]; width=min(288,len(rows))
    mae=np.mean(np.abs(pred-true),axis=0); chosen=np.argsort(mae)[[0,len(mae)//2,-1]]
    fig,axes=plt.subplots(2,2,figsize=(16,9),sharex=True); axes=axes.ravel()
    for ax,n,c in zip(axes[:3],chosen,COLORS):
        ax.plot(rows[:width],true[:width,n],color="#F7F7FF",lw=1.8,label="Observed"); ax.plot(rows[:width],pred[:width,n],color=c,lw=1.5,label="GCN+GRU")
        ax.set_title(f"Node {n} • test MAE {mae[n]:.2f}"); ax.grid(alpha=.2)
    axes[3].plot(rows[:width],true[:width].mean(1),color="#F7F7FF",lw=2,label="Observed mean"); axes[3].plot(rows[:width],pred[:width].mean(1),color=COLORS[0],lw=1.8,label="Predicted mean"); axes[3].set_title("Network-wide mean")
    for ax in axes: ax.legend(frameon=False); ax.set_ylabel("Speed (unit pending)")
    axes[2].set_xlabel("Original target row index"); axes[3].set_xlabel("Original target row index")
    fig.suptitle("Next-step Traffic Prediction • First 288 Test Targets",fontsize=21,weight="bold"); fig.tight_layout(rect=[0,.02,1,.94])
    return _save(fig,out,"figure_09_prediction_vs_observed")


def training_comparison(metrics: dict,out: Path):
    _style(); hist=pd.DataFrame(metrics["training"]["history"]); fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,9))
    ax1.plot(hist.epoch,hist.train_mse_standardized,color=COLORS[0],marker="o",label="Train MSE"); ax1.plot(hist.epoch,hist.validation_mse_standardized,color=COLORS[1],marker="o",label="Validation MSE")
    ax1.axvline(metrics["training"]["best_epoch"],color=COLORS[2],ls="--",label=f"Best epoch {metrics['training']['best_epoch']}"); ax1.set(title="Optimization History",xlabel="Epoch",ylabel="MSE (standardized scale)"); ax1.grid(alpha=.25); ax1.legend(frameon=False)
    labels=["MAE","RMSE"]; model=[metrics["test_metrics"]["mae"],metrics["test_metrics"]["rmse"]]; base=[metrics["persistence_baseline_test_metrics"]["mae"],metrics["persistence_baseline_test_metrics"]["rmse"]]
    x=np.arange(2); width=.34; ax2.bar(x-width/2,model,width,color=COLORS[0],label="GCN+GRU"); ax2.bar(x+width/2,base,width,color=COLORS[1],label="Persistence")
    ax2.set_xticks(x,labels); ax2.set(title="Held-out Test Performance",ylabel="Error (original speed scale)"); ax2.grid(axis="y",alpha=.25); ax2.legend(frameon=False)
    for i,v in enumerate(model): ax2.text(i-width/2,v,f"{v:.3f}",ha="center",va="bottom")
    for i,v in enumerate(base): ax2.text(i+width/2,v,f"{v:.3f}",ha="center",va="bottom")
    fig.suptitle("Training Convergence and Honest Baseline Comparison",fontsize=21,weight="bold"); fig.tight_layout(rect=[0,.02,1,.94])
    return _save(fig,out,"figure_10_training_and_baseline")


def make_step4_figures(root: Path) -> dict[str,list[str]]:
    out=root/"outputs"/"figures"; _,adjacency,_=load_traffic_graph(root)
    embedding=np.load(root/"outputs"/"embeddings"/"E_st.npy"); arrays=np.load(root/"outputs"/"metrics"/"st_evaluation_arrays.npz")
    metrics=json.loads((root/"outputs"/"metrics"/"st_training_metrics.json").read_text()); metadata=json.loads((root/"outputs"/"embeddings"/"E_st_metadata.json").read_text())
    return {"figure_05":architecture(out),"figure_06":propagation(adjacency,out),"figure_07":hidden_states(arrays["selected_hidden_sequence"],metadata,out),
            "figure_08":embedding_pca(embedding,out),"figure_09":prediction_plot(arrays,out),"figure_10":training_comparison(metrics,out)}

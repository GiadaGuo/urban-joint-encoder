"""PPT-ready Step 6 alignment and honest fusion visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.decomposition import PCA


COLORS=["#56CFE1","#FFB000","#F28482","#B983FF","#90BE6D"]


def _style():
    plt.rcParams.update({"figure.facecolor":"#08111F","axes.facecolor":"#0D1B2A","axes.edgecolor":"#7693B3","axes.labelcolor":"#E8F1FA","text.color":"#E8F1FA","xtick.color":"#C5D5E5","ytick.color":"#C5D5E5","grid.color":"#38516B","font.size":12,"axes.titleweight":"bold","axes.titlesize":19,"font.family":"sans-serif","font.sans-serif":["PingFang SC","Hiragino Sans GB","Arial Unicode MS","DejaVu Sans"],"axes.unicode_minus":False})


def _save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True); png=out/f"{stem}.png"; svg=out/f"{stem}.svg"; fig.savefig(png,dpi=300,facecolor=fig.get_facecolor()); fig.savefig(svg,facecolor=fig.get_facecolor()); plt.show(); plt.close(fig); return [str(png),str(svg)]


def architecture(out):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    boxes=[(.7,5.4,2.5,1.5,"E_st\n265 × 32","#184E77"),(.7,2.1,2.5,1.5,"E_se\n277 × 32","#7B2CBF"),(4.2,3.75,2.6,1.5,"Verified graph paths\nmask 265 × 277","#1A759F"),(7.8,5.4,2.8,1.5,"4-head Cross-Attention\nUNTRAINED DEMO","#A44A3F"),(7.8,2.1,2.8,1.5,"Traceable rule context\nfixed weights","#287271"),(11.7,3.75,3,1.5,"Gated residual + LayerNorm\nE_joint 265 × 32","#5A189A")]
    for x,y,w,h,label,c in boxes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",fc=c,ec="white",alpha=.88)); ax.text(x+w/2,y+h/2,label,ha="center",va="center",weight="bold",fontsize=14)
    for a,b in [((3.2,6.15),(4.2,4.9)),((3.2,2.85),(4.2,4.1)),((6.8,4.5),(7.8,6.15)),((6.8,4.5),(7.8,2.85)),((10.6,2.85),(11.7,4.25)),((10.6,6.15),(11.7,4.75))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=18,color="#E8F1FA",lw=2))
    ax.set_title("Spatiotemporal–Semantic Alignment and Fusion Architecture",pad=18)
    ax.text(8,.65,"Solid output E_joint uses deterministic rule fusion • Cross-Attention branch is forward-only architecture demonstration",ha="center",color="#FFCF70",fontsize=13)
    return _save(fig,out,"figure_17_cross_modal_architecture")


def mask_plot(mask,out):
    _style(); fig,ax=plt.subplots(figsize=(16,9)); image=ax.imshow(mask,aspect="auto",cmap="cividis",interpolation="nearest",vmin=0,vmax=1)
    ax.set(title="Real Relation-constrained Cross-modal Attention Mask",xlabel="Semantic node row in E_se (277 keys)",ylabel="Spatial grid node_index (265 queries)")
    ax.axhline(146,color="#F28482",lw=1.5,ls="--",label="Case grid 146"); ax.legend(frameon=False); cb=fig.colorbar(image,ax=ax,pad=.015,ticks=[0,1]); cb.ax.set_yticklabels(["Blocked","Allowed"])
    ax.text(.01,.02,f"{mask.sum():,} allowed / {mask.size:,} total = {mask.mean():.3%} • every row has ≥ {mask.sum(1).min()} valid keys",transform=ax.transAxes,color="#56CFE1")
    return _save(fig,out,"figure_18_relation_constrained_mask")


def paths_146(alignment,out):
    _style(); items=alignment["provenance_by_query_grid"]["146"]; categories={}
    for node,info in items.items():
        for p in info["paths"]: categories.setdefault(p["category"],[]).append(node)
    order=["SELF_SPATIAL_GRID","HAS_POI","ONE_HOP_ADJACENT_GRID","LOCAL_EVENT","EVENT_RELATION","EVENT_PATH_RULE"]
    fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.6,3.65),2.2,1.5,boxstyle="round,pad=.08",fc="#56CFE1",ec="white")); ax.text(1.7,4.4,"Query grid 146",ha="center",va="center",weight="bold",fontsize=16)
    ys=np.linspace(7.2,1.1,len(order))
    for idx,(cat,y) in enumerate(zip(order,ys)):
        nodes=categories.get(cat,[]); label=cat+"\n"+"\n".join(n.replace("poi_category:","POI-").replace("event:","evt:").replace("rule:","rule:") for n in nodes)
        ax.add_patch(FancyBboxPatch((5,y-.5),4.2,1,boxstyle="round,pad=.06",fc=COLORS[idx%len(COLORS)],ec="white",alpha=.85)); ax.text(7.1,y,label,ha="center",va="center",fontsize=10,weight="bold")
        ax.add_patch(FancyArrowPatch((2.8,4.4),(5,y),arrowstyle="-|>",mutation_scale=15,color="#7693B3",lw=1.5))
    ax.text(12,4.4,"9 allowed semantic keys\nEach path stored in\ncross_modal_alignment.json",ha="center",va="center",fontsize=15,color="#C5D5E5")
    ax.set_title("Traceable Semantic Association Paths for Case Grid 146",pad=18)
    ax.text(8,.35,"Event-path associations are graph relations, not causal explanations",ha="center",color="#F4B6B6")
    return _save(fig,out,"figure_19_grid146_semantic_paths")


def attention_heatmap(weights,alignment,out):
    _style(); items=alignment["provenance_by_query_grid"]["146"]; ordered=sorted(items.items(),key=lambda kv:kv[1]["semantic_row_index"]); rows=[v["semantic_row_index"] for _,v in ordered]; labels=[n.replace("poi_category:","POI-").replace("event:","evt:").replace("rule:","rule:") for n,_ in ordered]; values=weights[:,146,rows]
    fig,ax=plt.subplots(figsize=(16,9)); image=ax.imshow(values,aspect="auto",cmap="magma",interpolation="nearest"); ax.set_yticks(range(4),[f"Head {i+1}" for i in range(4)]); ax.set_xticks(range(len(labels)),labels,rotation=28,ha="right"); ax.set(title="Masked Cross-Attention Weights • Grid 146",xlabel="Allowed semantic keys only",ylabel="Attention head")
    for i in range(4):
        for j in range(len(labels)): ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=9,color="white" if values[i,j]<.45 else "black")
    cb=fig.colorbar(image,ax=ax,pad=.015); cb.set_label("Actual forward attention allocation")
    ax.text(.01,.02,"UNTRAINED ARCHITECTURE DEMO • actual PyTorch forward, fixed seed 42 • weights are not causal contributions",transform=ax.transAxes,color="#FFB000",weight="bold")
    fig.tight_layout()
    return _save(fig,out,"figure_20_untrained_attention_weights")


def representation_comparison(e_st,e_se_grid,e_joint,out):
    _style(); fig,axes=plt.subplots(1,3,figsize=(16,9)); datasets=[("E_st",e_st,COLORS[0]),("Aligned grid rows of E_se",e_se_grid,COLORS[1]),("E_joint rule baseline",e_joint,COLORS[2])]
    for ax,(name,data,color) in zip(axes,datasets):
        norms=np.linalg.norm(data,axis=1); low,high=float(norms.min()),float(norms.max()); pad=max((high-low)*.03,1e-5); bins=np.linspace(low-pad,high+pad,20)
        ax.hist(norms,bins=bins,color=color,alpha=.85); ax.axvline(norms[146],color="white",ls="--",lw=2,label=f"Grid 146: {norms[146]:.2f}"); ax.set(title=name,xlabel="Node-vector L2 norm",ylabel="Grid count"); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Representation Diagnostics • Same 265 Verified Grid Identities",fontsize=21,weight="bold"); fig.tight_layout(rect=[0,.04,1,.94]); fig.text(.5,.025,"Descriptive comparison only • different encoders have different latent coordinate systems • no performance claim",ha="center",color="#FFB000")
    return _save(fig,out,"figure_21_representation_comparison")


def joint_embedding(e_joint,out):
    _style(); pca=PCA(2); xy=pca.fit_transform(e_joint); norm=np.linalg.norm(e_joint,axis=1); fig,ax=plt.subplots(figsize=(16,9)); s=ax.scatter(xy[:,0],xy[:,1],c=norm,cmap="viridis",s=54,alpha=.88,edgecolors="white",linewidths=.35)
    ax.scatter(xy[146,0],xy[146,1],s=260,facecolors="none",edgecolors="#F28482",linewidths=2.5,label="Case grid 146"); ax.annotate("146",xy[146],xytext=(7,7),textcoords="offset points",color="#F28482",weight="bold")
    cb=fig.colorbar(s,ax=ax,pad=.015); cb.set_label("E_joint L2 norm"); ax.set(title="Joint State Node Embeddings • PCA of Rule-fusion Baseline",xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"); ax.grid(alpha=.2); ax.legend(frameon=False)
    ax.text(.01,.02,"DETERMINISTIC RULE BASELINE • not trained Cross-Attention • matrix node indices, not geographic coordinates",transform=ax.transAxes,color="#FFB000",weight="bold")
    return _save(fig,out,"figure_22_ejoint_embedding_pca")


def make_step6_figures(root:Path):
    out=root/"outputs/figures"; alignment=json.loads((root/"outputs/alignment/cross_modal_alignment.json").read_text()); se_meta=json.loads((root/"outputs/embeddings/E_se_metadata.json").read_text()); mask=np.load(root/"outputs/alignment/attention_mask.npy"); demo=np.load(root/"outputs/alignment/untrained_attention_forward.npz"); e_st=np.load(root/"outputs/embeddings/E_st.npy"); e_se=np.load(root/"outputs/embeddings/E_se.npy"); e_joint=np.load(root/"outputs/embeddings/E_joint.npy"); e_se_grid=np.stack([e_se[se_meta["node_id_to_row_index"][f"grid:{i}"]] for i in range(265)])
    return {"figure_17":architecture(out),"figure_18":mask_plot(mask,out),"figure_19":paths_146(alignment,out),"figure_20":attention_heatmap(demo["attention_weights"],alignment,out),"figure_21":representation_comparison(e_st,e_se_grid,e_joint,out),"figure_22":joint_embedding(e_joint,out)}

"""Reusable, read-only data audit and visualization utilities for Step 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


POI_NAMES = [
    "交通设施服务",
    "住宿服务",
    "公司企业",
    "医疗保健服务",
    "购物服务",
    "风景名胜",
    "餐饮服务",
]


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root when called from either root or notebooks/."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Cannot locate project root containing pyproject.toml and data/")


def read_numeric_csv(path: Path) -> pd.DataFrame:
    """Read a headerless numeric CSV, preserving its first observation."""
    frame = pd.read_csv(path, header=None, encoding="utf-8-sig")
    return frame.apply(pd.to_numeric, errors="raise")


def load_all(root: Path) -> dict[str, object]:
    base = root / "data" / "raw" / "Main data" / "prediction_data"
    paths = {
        "traffic": base / "feature_matrix_X.csv",
        "adjacency": base / "adj_matrix.csv",
        **{f"poi_{i}": base / "POI" / f"poi{i}.csv" for i in range(1, 8)},
        "precipitation_station_1": base / "precipitation" / "station1.csv",
        "precipitation_station_2": base / "precipitation" / "station2.csv",
    }
    frames = {name: read_numeric_csv(path) for name, path in paths.items()}
    return {"base": base, "paths": paths, "frames": frames}


def frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    missing = int(np.isnan(values).sum())
    count = int(values.size)
    return {
        "shape": [int(x) for x in frame.shape],
        "dtypes": sorted({str(x) for x in frame.dtypes}),
        "missing_count": missing,
        "missing_ratio": float(missing / count) if count else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "zero_ratio": float(np.count_nonzero(values == 0) / count) if count else None,
    }


def build_audit(bundle: dict[str, object], root: Path) -> dict[str, object]:
    frames = bundle["frames"]
    paths = bundle["paths"]
    traffic = frames["traffic"].to_numpy(float)
    adjacency = frames["adjacency"].to_numpy(float)
    poi = np.vstack([frames[f"poi_{i}"].to_numpy(float).ravel() for i in range(1, 8)]).T
    precip = [frames[f"precipitation_station_{i}"].to_numpy(float).ravel() for i in (1, 2)]
    square = adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]
    symmetric = bool(square and np.allclose(adjacency, adjacency.T, equal_nan=True))
    graph = nx.from_numpy_array(adjacency) if square else nx.Graph()
    self_loop_nodes = np.flatnonzero(np.diag(adjacency) != 0).astype(int).tolist() if square else []
    isolated = list(nx.isolates(graph))
    non_binary = np.argwhere(~np.isin(adjacency, [0, 1]))
    rel = lambda p: p.relative_to(root).as_posix()

    poi_node_aligned = bool(all(frames[f"poi_{i}"].size == traffic.shape[1] for i in range(1, 8)))
    precip_length_aligned = bool(all(len(x) == traffic.shape[0] for x in precip))
    audit = {
        "generated_by": "notebooks/01_data_overview.ipynb",
        "read_policy": "All supplied CSV files are read with header=None and UTF-8 BOM handling.",
        "files": {
            name: {"path": rel(paths[name]), **frame_summary(frame)}
            for name, frame in frames.items()
        },
        "traffic": {"time_steps": int(traffic.shape[0]), "spatial_nodes": int(traffic.shape[1])},
        "poi": {
            "category_count": 7,
            "category_names": POI_NAMES,
            "matrix_shape_nodes_by_categories": [int(x) for x in poi.shape],
            "nodes_with_any_poi": int(np.count_nonzero(poi.sum(axis=1) > 0)),
            "nodes_without_any_poi": int(np.count_nonzero(poi.sum(axis=1) == 0)),
            "coverage_ratio": float(np.mean(poi.sum(axis=1) > 0)),
            "nonzero_nodes_by_category": {
                name: int(np.count_nonzero(poi[:, i])) for i, name in enumerate(POI_NAMES)
            },
        },
        "precipitation": {
            "series_lengths": [int(len(x)) for x in precip],
            "station_labels_from_source": ["徐家汇", "浦东"],
        },
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "is_square": bool(square),
            "is_symmetric": symmetric,
            "self_loop_count": int(len(self_loop_nodes)),
            "self_loop_nodes": self_loop_nodes,
            "isolated_node_count": int(len(isolated)),
            "isolated_nodes": [int(x) for x in isolated],
            "non_binary_entry_count": int(len(non_binary)),
        },
        "alignment_checks": {
            "traffic_columns_equal_adjacency_nodes": bool(traffic.shape[1] == adjacency.shape[0] == adjacency.shape[1]),
            "poi_lengths_equal_traffic_nodes": poi_node_aligned,
            "precipitation_lengths_equal_traffic_steps": precip_length_aligned,
            "node_identity_order_verified": False,
            "time_identity_verified": False,
            "assessment": (
                "Shape/length compatibility passes. The generator code indicates shared grid-column order for traffic, "
                "adjacency and POI, but exported ID labels are absent; supplied files alone cannot re-verify identity. "
                "Precipitation is positionally compatible, but timestamps were removed and semantic time alignment is unverified."
            ),
        },
        "source_findings": {
            "confirmed": [
                "Traffic rows are consecutive batch slots and columns are selected Huangpu District honeycomb grids.",
                "The source reshapes 8640 slots as 30 days x 288 slots/day, implying five-minute slots.",
                "Traffic values are local_avg_speed after tensor completion; values above 40 were marked missing before completion.",
                "Adjacency is road-based: roads within grids connected by a road touch relation induce grid edges.",
                "Disconnected grids were linked to a connected grid through a honeycomb adjacency shortest path (1..10 hops).",
                "POI files are per-grid counts for the seven named categories and were exported in traffic grid-column order.",
                "Precipitation files retain the precipitation-mm column from Xujiahui and Pudong station CSVs.",
            ],
            "pending_confirmation": [
                "Original honeycomb/FID labels are absent from the headerless exports, so node identity/order cannot be independently audited.",
                "No timestamps are present in traffic or precipitation exports, so exact temporal alignment cannot be independently audited.",
                "No geographic coordinates are present in prediction_data; topology plots are not maps.",
                "Traffic-speed physical unit is not stated in the inspected source or README.",
                "The precipitation notebook does not document source sampling interval or explicit resampling/alignment.",
            ],
        },
        "known_reference_code_risk": (
            "Reference load_features() uses pd.read_csv(feat_path) without header=None, which treats the first numeric row "
            "as a header. This audit preserves all 8640 rows."
        ),
    }
    return audit


def audit_tables(audit: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, item in audit["files"].items():
        rows.append({"dataset": name, **{k: v for k, v in item.items() if k != "path"}})
    checks = pd.DataFrame(
        [(key, value) for key, value in audit["alignment_checks"].items() if key != "assessment"],
        columns=["check", "result"],
    )
    return pd.DataFrame(rows), checks


def save_json(audit: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#08111F", "axes.facecolor": "#0D1B2A", "axes.edgecolor": "#7693B3",
        "axes.labelcolor": "#E8F1FA", "text.color": "#E8F1FA", "xtick.color": "#C5D5E5",
        "ytick.color": "#C5D5E5", "grid.color": "#38516B", "font.size": 11,
        "axes.titleweight": "bold", "axes.titlesize": 18, "savefig.facecolor": "#08111F",
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def _save(fig: plt.Figure, out: Path, stem: str) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    png, svg = out / f"{stem}.png", out / f"{stem}.svg"
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return png, svg


def figure_01(traffic: np.ndarray, out: Path):
    _style()
    time_idx = np.unique(np.linspace(0, traffic.shape[0] - 1, 720, dtype=int))
    node_idx = np.arange(traffic.shape[1])
    sampled = traffic[np.ix_(time_idx, node_idx)].T
    fig, ax = plt.subplots(figsize=(16, 9))
    image = ax.imshow(sampled, aspect="auto", cmap="cividis", interpolation="nearest",
                      vmin=np.nanpercentile(traffic, 1), vmax=np.nanpercentile(traffic, 99))
    ax.set(title="Traffic Speed Across Space and Time", xlabel="Time step (720 evenly sampled from 8,640)", ylabel="Spatial node index")
    cb = fig.colorbar(image, ax=ax, pad=.015); cb.set_label("Traffic speed (unit: pending confirmation)")
    ax.text(.995, 1.015, "REAL OBSERVATIONS • P1–P99 COLOR RANGE", transform=ax.transAxes, ha="right", color="#56CFE1", fontsize=10)
    return _save(fig, out, "figure_01_traffic_speed_heatmap")


def figure_02(adjacency: np.ndarray, out: Path):
    _style(); graph = nx.from_numpy_array(adjacency)
    degree_rank = sorted(graph.degree, key=lambda x: (-x[1], x[0]))
    seeds = [node for node, _ in degree_rank[:8]]
    distances = nx.multi_source_dijkstra_path_length(graph, seeds, cutoff=2, weight=None)
    chosen = sorted(distances, key=lambda n: (distances[n], -graph.degree[n], n))[:80]
    sub = graph.subgraph(chosen).copy(); pos = nx.spring_layout(sub, seed=42, k=.42, iterations=150)
    degrees = np.array([sub.degree[n] for n in sub.nodes])
    fig, ax = plt.subplots(figsize=(16, 9)); ax.set_axis_off()
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#5B7896", alpha=.42, width=.8)
    nodes = nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=38 + 16 * degrees, node_color=degrees,
                                   cmap="plasma", edgecolors="#E8F1FA", linewidths=.45)
    cb=fig.colorbar(nodes, ax=ax, shrink=.72, pad=.01); cb.set_label("Degree within displayed subgraph")
    ax.set_title("Road-derived Spatial Adjacency • 80-node Topology View", pad=18)
    ax.text(.5, .015, "Spring layout (seed=42) • positions are topological, not geographic", transform=ax.transAxes,
            ha="center", color="#56CFE1", fontsize=12)
    return _save(fig, out, "figure_02_spatial_adjacency_topology")


def figure_03(poi: np.ndarray, out: Path):
    _style(); order = np.argsort(poi.sum(axis=1)); shown = poi[order].T
    vmax = max(1, float(np.nanpercentile(shown, 99)))
    fig, ax = plt.subplots(figsize=(16, 9))
    image=ax.imshow(shown, aspect="auto", cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set(title="POI Semantic Profile Across 265 Spatial Nodes", xlabel="Nodes ordered by total POI count", ylabel="POI category")
    ax.set_yticks(range(7), POI_NAMES); ax.set_xticks(np.linspace(0, 264, 6), [f"{x+1}" for x in np.linspace(0,264,6,dtype=int)])
    cb=fig.colorbar(image, ax=ax, pad=.015); cb.set_label("POI count (color capped at P99)")
    ax.text(.995, 1.015, "SOURCE-CONFIRMED CATEGORY ORDER", transform=ax.transAxes, ha="right", color="#56CFE1", fontsize=10)
    return _save(fig, out, "figure_03_poi_semantic_distribution")


def figure_04(traffic: np.ndarray, out: Path):
    _style(); variability=np.nanstd(traffic, axis=0); means=np.nanmean(traffic, axis=0)
    candidates=[int(np.nanargmin(variability)), int(np.nanargmax(variability)), int(np.nanargmin(means)), int(np.nanargmax(means))]
    chosen=[]
    for n in candidates:
        if n not in chosen: chosen.append(n)
    start, width = 0, min(576, traffic.shape[0]); x=np.arange(start,start+width)
    colors=["#56CFE1", "#FFB000", "#B983FF", "#F28482"]
    fig, ax=plt.subplots(figsize=(16,9))
    for n,c in zip(chosen,colors): ax.plot(x,traffic[start:start+width,n],lw=1.15,alpha=.82,color=c,label=f"Node {n} (μ={means[n]:.1f}, σ={variability[n]:.1f})")
    ax.plot(x,np.nanmean(traffic[start:start+width],axis=1),color="#F7F7FF",lw=2.5,label="Global mean")
    ax.set(title="Observed Traffic State Evolution • First 48 Hours", xlabel="Time step (5-minute slots inferred from source batching)", ylabel="Traffic speed (unit: pending confirmation)")
    ax.grid(alpha=.25); ax.legend(ncol=3,frameon=False,loc="upper right")
    ax.text(.005,.02,"Representative nodes selected by observed mean and variability — no synthetic events",transform=ax.transAxes,color="#56CFE1",fontsize=10)
    return _save(fig,out,"figure_04_traffic_state_evolution"), chosen


def make_all_figures(bundle: dict[str, object], root: Path) -> dict[str, object]:
    f=bundle["frames"]; out=root/"outputs"/"figures"
    traffic=f["traffic"].to_numpy(float); adj=f["adjacency"].to_numpy(float)
    poi=np.vstack([f[f"poi_{i}"].to_numpy(float).ravel() for i in range(1,8)]).T
    p1=figure_01(traffic,out); p2=figure_02(adj,out); p3=figure_03(poi,out); p4,chosen=figure_04(traffic,out)
    return {"figure_01":p1,"figure_02":p2,"figure_03":p3,"figure_04":p4,"representative_nodes":chosen}

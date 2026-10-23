"""Step 8.2: unified, inference-only evaluation for ST-only and Joint.

Run from the project root:
    uv run --no-sync python -m urban_joint_encoder.evaluate_joint --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from urban_joint_encoder.causal_semantic_snapshots import CausalSemanticSnapshotBuilder
from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.data_loader import (
    SplitConfig, Standardizer, fit_train_standardizer,
    load_traffic_graph, split_boundaries,
)
from urban_joint_encoder.joint_dataset import JointTrafficDataset, joint_collate
from urban_joint_encoder.models.joint_encoder import JointSpatiotemporalSemanticEncoder
from urban_joint_encoder.models.st_encoder import STForecastModel


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - truth.astype(np.float64)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error * error)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--case-row", type=int, default=7347)
    parser.add_argument("--case-grid", type=int, default=146)
    args = parser.parse_args()

    root = find_project_root().resolve()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.set_grad_enabled(False)

    started = time.perf_counter()
    traffic, adjacency, _ = load_traffic_graph(root)
    bounds = split_boundaries(len(traffic), SplitConfig())
    st_ckpt_path = root / "outputs/models/st_encoder_best.pt"
    joint_ckpt_path = root / "outputs/models/joint_full_best.pt"
    st_ckpt = torch.load(st_ckpt_path, map_location="cpu", weights_only=True)
    joint_ckpt = torch.load(joint_ckpt_path, map_location="cpu", weights_only=True)
    st_cfg = st_ckpt["config"]
    joint_cfg = joint_ckpt["config"]

    if int(st_cfg["sequence_length"]) != int(joint_cfg["sequence_length"]):
        raise ValueError("ST and Joint sequence lengths differ")
    sequence_length = int(joint_cfg["sequence_length"])
    scaler = Standardizer(**st_ckpt["scaler"])
    expected_scaler = fit_train_standardizer(traffic, bounds["train"][1])
    if not (np.isclose(scaler.mean, expected_scaler.mean, rtol=0, atol=1e-6) and np.isclose(scaler.std, expected_scaler.std, rtol=0, atol=1e-6)):
        raise ValueError("ST checkpoint scaler differs from train-only fitted scaler")
    normalized = scaler.transform(traffic).astype(np.float32)

    st_model = STForecastModel(torch.from_numpy(adjacency), int(st_cfg["hidden_dim"])).to(device)
    st_model.load_state_dict(st_ckpt["model_state_dict"], strict=True)
    st_model.eval()
    joint_model = JointSpatiotemporalSemanticEncoder(
        torch.from_numpy(adjacency), st_ckpt["model_state_dict"],
        int(joint_cfg["hidden_dim"]), 6,
    ).to(device)
    joint_model.load_state_dict(joint_ckpt["model_state_dict"], strict=True)
    joint_model.eval()

    builder = CausalSemanticSnapshotBuilder(
        root, int(joint_cfg["recent_event_horizon"]), int(joint_cfg["max_dynamic_events"])
    )
    snapshot_cache: dict[int, object] = {}
    def get_snapshots(rows):
        result = []
        for row in rows:
            row = int(row)
            if row not in snapshot_cache:
                snapshot_cache[row] = builder.build_semantic_snapshot(row)
            result.append(snapshot_cache[row].to(device))
        return result

    chosen_splits = ["validation", "test"] if args.split == "both" else [args.split]
    comparison: list[dict] = []
    report_splits = {}
    case_found = False
    all_predictions: dict[str, dict] = {}
    for split in chosen_splits:
        lo, hi = bounds[split]
        rows = list(range(lo, hi))
        dataset = JointTrafficDataset(normalized, rows, sequence_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=joint_collate)
        collected = {"ST-only": [], "Joint": [], "Persistence": []}
        target_parts = []
        target_rows = []
        print(f"Evaluating {split}: rows [{lo}, {hi}), {len(dataset)} windows", flush=True)
        for batch_idx, (x, y, row_batch) in enumerate(loader, 1):
            x_gpu = x.to(device)
            st_norm, _ = st_model(x_gpu)
            joint_norm = joint_model(x_gpu, get_snapshots(row_batch))["prediction"]
            st_raw = scaler.inverse(st_norm.cpu().numpy())
            joint_raw = scaler.inverse(joint_norm.cpu().numpy())
            truth_raw = scaler.inverse(y.numpy())
            persistence_raw = np.stack([traffic[r - 1] for r in row_batch])
            collected["ST-only"].append(st_raw)
            collected["Joint"].append(joint_raw)
            collected["Persistence"].append(persistence_raw)
            target_parts.append(truth_raw)
            target_rows.extend(row_batch)
            if batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"  {split}: {batch_idx}/{len(loader)} batches", flush=True)
        targets = np.concatenate(target_parts)
        predictions = {name: np.concatenate(parts) for name, parts in collected.items()}
        row_array = np.asarray(target_rows, dtype=np.int64)
        if not np.array_equal(row_array, np.asarray(rows)):
            raise AssertionError("Prediction rows are not in chronological order")
        split_metrics = {}
        for name in ("Persistence", "ST-only", "Joint"):
            values = metrics(predictions[name], targets)
            split_metrics[name] = values
            comparison.append({"split": split, "model": name, "windows": len(rows),
                               "num_grids": int(traffic.shape[1]), **values})
            print(f"  {name}: MAE={values['mae']:.6f} RMSE={values['rmse']:.6f}", flush=True)
        report_splits[split] = {"target_rows_half_open": [lo, hi], "metrics": split_metrics}
        all_predictions[split] = {"rows": row_array, "targets": targets, "predictions": predictions}

        if args.case_row in rows:
            i = args.case_row - lo
            case_rows = []
            for grid in range(traffic.shape[1]):
                case_rows.append({
                    "target_row": args.case_row, "grid_index": grid,
                    "is_selected_grid": int(grid == args.case_grid),
                    "target_raw": float(targets[i, grid]),
                    "persistence_raw": float(predictions["Persistence"][i, grid]),
                    "st_only_raw": float(predictions["ST-only"][i, grid]),
                    "joint_raw": float(predictions["Joint"][i, grid]),
                    "st_abs_error": float(abs(predictions["ST-only"][i, grid] - targets[i, grid])),
                    "joint_abs_error": float(abs(predictions["Joint"][i, grid] - targets[i, grid])),
                })
            write_csv(root / f"outputs/predictions/prediction_case_{args.case_row}.csv",
                      case_rows, list(case_rows[0].keys()))
            case_found = True

    out = root / "outputs/metrics"
    write_csv(out / "model_comparison.csv", comparison,
              ["split", "model", "windows", "num_grids", "mae", "rmse"])
    report = {
        "stage": "Step 8.2 unified evaluation (inference only)",
        "model_checkpoints": {"ST-only": str(st_ckpt_path.relative_to(root)),
                              "Joint": str(joint_ckpt_path.relative_to(root))},
        "joint_checkpoint_best_epoch": joint_ckpt.get("epoch"),
        "joint_checkpoint_validation_rmse_raw": joint_ckpt.get("validation_rmse_raw"),
        "device": str(device), "torch_version": torch.__version__,
        "standardizer": scaler.as_dict(),
        "split_boundaries_half_open": {k: list(v) for k, v in bounds.items()},
        "splits": report_splits,
        "test_set_used_for_training_or_model_selection": False,
        "training_window_counts": {"ST-only": 6036, "Joint": 4031},
        "comparison_limitation": "Same evaluation targets and scaler; different training window counts. Not a controlled ablation.",
        "case": {"target_row": args.case_row, "grid_index": args.case_grid,
                 "exported": case_found, "grid_index_is_not_geographic_fid": True,
                 "real_timestamp_unavailable": True},
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(out / "evaluation_report.json", report)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        names = ["Persistence", "ST-only", "Joint"]
        colors = ["#737373", "#2563eb", "#d97706"]
        x = np.arange(len(names))
        for ax, metric_name in zip(axes, ("mae", "rmse")):
            width = 0.35 if len(chosen_splits) == 2 else 0.6
            for j, split in enumerate(chosen_splits):
                vals = [report_splits[split]["metrics"][name][metric_name] for name in names]
                offsets = (j - (len(chosen_splits) - 1) / 2) * width
                bars = ax.bar(x + offsets, vals, width, label=split,
                              color=colors if j == 0 else ["#a3a3a3", "#93c5fd", "#fdba74"],
                              edgecolor="#262626", linewidth=0.5)
                for bar, val in zip(bars, vals):
                    ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width()/2, val),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, names)
            ax.set_title(metric_name.upper() + " (raw value scale)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False)
        fig.suptitle("Unified traffic forecasting evaluation", fontsize=13)
        figure_path = root / "outputs/figures/model_comparison.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {figure_path}", flush=True)
    except ImportError:
        print("matplotlib unavailable; CSV and JSON were saved, PNG skipped", flush=True)

    print("Saved outputs/metrics/model_comparison.csv", flush=True)
    print("Saved outputs/metrics/evaluation_report.json", flush=True)
    print(f"Case row {args.case_row} exported: {case_found}", flush=True)
    print(f"Evaluation finished in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

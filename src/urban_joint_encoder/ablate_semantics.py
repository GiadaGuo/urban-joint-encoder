"""Step 8.3: inference-only dynamic-event attention masking sensitivity test.

Run: uv run --no-sync python -m urban_joint_encoder.ablate_semantics --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from urban_joint_encoder.causal_semantic_snapshots import CausalSemanticSnapshotBuilder
from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.data_loader import SplitConfig, Standardizer, fit_train_standardizer, load_traffic_graph, split_boundaries
from urban_joint_encoder.joint_dataset import JointTrafficDataset, joint_collate
from urban_joint_encoder.models.joint_encoder import JointSpatiotemporalSemanticEncoder
from urban_joint_encoder.evaluate_joint import metrics, save_json, write_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--case-row', type=int, default=7347)
    args = p.parse_args()
    root = find_project_root().resolve()
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(device_name)
    start = time.perf_counter()

    traffic, adjacency, _ = load_traffic_graph(root)
    bounds = split_boundaries(len(traffic), SplitConfig())
    st_ckpt = torch.load(root / 'outputs/models/st_encoder_best.pt', map_location='cpu', weights_only=True)
    joint_ckpt = torch.load(root / 'outputs/models/joint_full_best.pt', map_location='cpu', weights_only=True)
    cfg = joint_ckpt['config']
    scaler = Standardizer(**st_ckpt['scaler'])
    fitted = fit_train_standardizer(traffic, bounds['train'][1])
    if not (np.isclose(scaler.mean, fitted.mean, atol=1e-6, rtol=0) and np.isclose(scaler.std, fitted.std, atol=1e-6, rtol=0)):
        raise ValueError('Checkpoint scaler differs from training-only scaler')
    if int(st_ckpt['config']['sequence_length']) != int(cfg['sequence_length']):
        raise ValueError('Sequence length mismatch')
    normalized = scaler.transform(traffic).astype(np.float32)
    model = JointSpatiotemporalSemanticEncoder(
        torch.from_numpy(adjacency), st_ckpt['model_state_dict'], int(cfg['hidden_dim']), 6
    ).to(device)
    model.load_state_dict(joint_ckpt['model_state_dict'], strict=True)
    model.eval()
    builder = CausalSemanticSnapshotBuilder(root, int(cfg['recent_event_horizon']), int(cfg['max_dynamic_events']))

    lo, hi = bounds['validation']
    rows = list(range(lo, hi))
    dataset = JointTrafficDataset(normalized, rows, int(cfg['sequence_length']))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=joint_collate)
    full_parts, masked_parts, target_parts = [], [], []
    event_counts = []
    event_attention_connections = 0
    cache = {}
    print(f'Validation dynamic-attention sensitivity: [{lo}, {hi}), {len(rows)} windows', flush=True)

    with torch.inference_mode():
        for batch_i, (x, y, row_batch) in enumerate(loader, 1):
            full_snapshots, masked_snapshots = [], []
            for row in row_batch:
                row = int(row)
                if row not in cache:
                    snapshot = builder.build_semantic_snapshot(row)
                    event_idx = [i for i, typ in enumerate(snapshot.node_types) if typ == 'CongestionEvent']
                    mask = snapshot.attention_mask.clone()
                    if event_idx:
                        event_attention_connections += int(mask[:, event_idx].sum().item())
                        mask[:, event_idx] = False
                    if not mask.any(dim=1).all():
                        raise RuntimeError(f'Row {row}: masking events leaves a query without a semantic key')
                    cache[row] = (snapshot, replace(snapshot, attention_mask=mask), len(event_idx))
                original, masked, n_events = cache[row]
                full_snapshots.append(original.to(device))
                masked_snapshots.append(masked.to(device))
                event_counts.append(n_events)
            x = x.to(device)
            full = model(x, full_snapshots)['prediction']
            masked = model(x, masked_snapshots)['prediction']
            full_parts.append(scaler.inverse(full.cpu().numpy()))
            masked_parts.append(scaler.inverse(masked.cpu().numpy()))
            target_parts.append(scaler.inverse(y.numpy()))
            if batch_i % 25 == 0 or batch_i == len(loader):
                print(f'  {batch_i}/{len(loader)} batches', flush=True)

    truth = np.concatenate(target_parts)
    full = np.concatenate(full_parts)
    masked = np.concatenate(masked_parts)
    full_metrics, masked_metrics = metrics(full, truth), metrics(masked, truth)
    delta = {k: masked_metrics[k] - full_metrics[k] for k in ('mae', 'rmse')}
    print(f"Full Joint:   MAE={full_metrics['mae']:.6f} RMSE={full_metrics['rmse']:.6f}", flush=True)
    print(f"Mask Dynamic: MAE={masked_metrics['mae']:.6f} RMSE={masked_metrics['rmse']:.6f}", flush=True)
    print(f"Mask - Full:  dMAE={delta['mae']:+.6f} dRMSE={delta['rmse']:+.6f}", flush=True)
    print(f'Event nodes per snapshot: mean={np.mean(event_counts):.2f}, min={min(event_counts)}, max={max(event_counts)}', flush=True)

    metrics_dir = root / 'outputs/metrics'
    write_csv(metrics_dir / 'semantic_attention_sensitivity.csv', [
        {'condition': name, 'split': 'validation', 'windows': len(rows), 'num_grids': truth.shape[1], **values}
        for name, values in [('Full Joint', full_metrics), ('Mask Dynamic', masked_metrics)]
    ], ['condition', 'split', 'windows', 'num_grids', 'mae', 'rmse'])
    report = {
        'stage': 'Step 8.3 inference-only dynamic-event attention masking sensitivity',
        'split': 'validation', 'target_rows_half_open': [lo, hi], 'windows': len(rows),
        'checkpoint': 'outputs/models/joint_full_best.pt', 'device': str(device),
        'conditions': {'Full Joint': full_metrics, 'Mask Dynamic': masked_metrics},
        'mask_minus_full': delta,
        'event_nodes_per_snapshot': {'mean': float(np.mean(event_counts)), 'min': min(event_counts), 'max': max(event_counts)},
        'originally_allowed_event_attention_connections': event_attention_connections,
        'masking_method': 'Set attention_mask[:, CongestionEvent node indices] = False; graph node features, edges, semantic GNN, and model weights unchanged.',
        'limitation': 'Dynamic event messages can still propagate to static semantic nodes in the GNN; not complete event removal or retrained ablation. Test split not used.',
        'elapsed_seconds': time.perf_counter() - start,
    }
    save_json(metrics_dir / 'semantic_attention_sensitivity.json', report)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4), constrained_layout=True)
        for ax, metric in zip(axes, ('mae', 'rmse')):
            values = [full_metrics[metric], masked_metrics[metric]]
            bars = ax.bar(['Full Joint', 'Mask Dynamic'], values, color=['#2563eb', '#f59e0b'], edgecolor='#222222', linewidth=0.7)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            ax.set_title(metric.upper() + ' (validation, lower is better)')
            ax.set_ylim(0, max(values) * 1.2 if max(values) else 1)
            ax.grid(axis='y', alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle('Dynamic-event attention masking sensitivity (frozen Joint model)', fontsize=12)
        path = root / 'outputs/figures/semantic_attention_sensitivity.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path.relative_to(root)}', flush=True)
    except ImportError:
        print('matplotlib unavailable; CSV/JSON saved, PNG skipped', flush=True)
    print('Saved outputs/metrics/semantic_attention_sensitivity.csv', flush=True)
    print('Saved outputs/metrics/semantic_attention_sensitivity.json', flush=True)
    print(f'Finished in {report["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()

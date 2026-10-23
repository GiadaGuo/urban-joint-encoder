"""CPU training and artifact export for the Step 4 spatiotemporal encoder."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_joint_encoder.data_audit import find_project_root
from urban_joint_encoder.data_loader import SplitConfig, fit_train_standardizer, load_traffic_graph, split_boundaries
from urban_joint_encoder.models import STForecastModel
from urban_joint_encoder.temporal_dataset import TrafficWindowDataset


@dataclass
class TrainingConfig:
    seed: int = 42
    sequence_length: int = 12
    hidden_dim: int = 32
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 15
    patience: int = 4
    min_delta: float = 1e-5
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training); total, count = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x)
        loss = criterion(prediction, y)
        if training:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss.detach()) * x.shape[0]; count += x.shape[0]
    return total / count


@torch.no_grad()
def _predict(model, loader, scaler, device):
    model.eval(); predictions=[]; targets=[]; rows=[]
    for x, y, row in loader:
        prediction, _ = model(x.to(device))
        predictions.append(prediction.cpu().numpy()); targets.append(y.numpy()); rows.append(row.numpy())
    pred_norm=np.concatenate(predictions); target_norm=np.concatenate(targets)
    return scaler.inverse(pred_norm), scaler.inverse(target_norm), np.concatenate(rows)


def run_experiment(root: Path | None = None, config: TrainingConfig | None = None) -> dict[str, object]:
    root = (root or find_project_root()).resolve(); config = config or TrainingConfig()
    set_seed(config.seed); torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    started=time.perf_counter(); traffic, adjacency, source_paths=load_traffic_graph(root)
    split_config=SplitConfig(config.sequence_length, config.train_ratio, config.validation_ratio, config.test_ratio)
    bounds=split_boundaries(len(traffic), split_config); scaler=fit_train_standardizer(traffic, bounds["train"][1])
    normalized=scaler.transform(traffic).astype(np.float32)
    datasets={
        "train": TrafficWindowDataset(normalized, config.sequence_length, bounds["train"][1], config.sequence_length),
        "validation": TrafficWindowDataset(normalized, bounds["validation"][0], bounds["validation"][1], config.sequence_length),
        "test": TrafficWindowDataset(normalized, bounds["test"][0], bounds["test"][1], config.sequence_length),
    }
    generator=torch.Generator().manual_seed(config.seed)
    loaders={
        "train": DataLoader(datasets["train"],batch_size=config.batch_size,shuffle=True,generator=generator,num_workers=0),
        "validation": DataLoader(datasets["validation"],batch_size=config.batch_size,shuffle=False,num_workers=0),
        "test": DataLoader(datasets["test"],batch_size=config.batch_size,shuffle=False,num_workers=0),
    }
    device=torch.device(config.device); model=STForecastModel(torch.from_numpy(adjacency),config.hidden_dim).to(device)
    # Required forward contract test before training.
    with torch.no_grad():
        test_x=torch.zeros(2,config.sequence_length,traffic.shape[1],1,device=device)
        test_prediction,test_embedding=model(test_x)
    assert tuple(test_embedding.shape)==(2,traffic.shape[1],config.hidden_dim)
    assert tuple(test_prediction.shape)==(2,traffic.shape[1])

    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate); criterion=nn.MSELoss()
    output_models=root/"outputs"/"models"; output_metrics=root/"outputs"/"metrics"; output_embeddings=root/"outputs"/"embeddings"
    for folder in (output_models,output_metrics,output_embeddings): folder.mkdir(parents=True,exist_ok=True)
    checkpoint_path=output_models/"st_encoder_best.pt"; history=[]; best=float("inf"); stale=0; best_epoch=0
    for epoch in range(1,config.max_epochs+1):
        train_loss=_run_epoch(model,loaders["train"],criterion,device,optimizer)
        with torch.no_grad(): validation_loss=_run_epoch(model,loaders["validation"],criterion,device)
        history.append({"epoch":epoch,"train_mse_standardized":train_loss,"validation_mse_standardized":validation_loss})
        print(f"epoch={epoch:02d} train={train_loss:.6f} validation={validation_loss:.6f}")
        if validation_loss < best-config.min_delta:
            best=validation_loss; stale=0; best_epoch=epoch
            torch.save({"model_state_dict":model.state_dict(),"config":asdict(config),"scaler":scaler.as_dict(),"adjacency_shape":list(adjacency.shape)},checkpoint_path)
        else:
            stale+=1
            if stale>=config.patience: break

    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    pred,target,target_rows=_predict(model,loaders["test"],scaler,device)
    model_metrics=_metrics(pred,target)
    baseline=np.stack([traffic[row-1] for row in target_rows])
    baseline_metrics=_metrics(baseline,target)

    # First chronological real test sample is the reproducible fusion hand-off.
    x0,y0,row0=datasets["test"][0]
    with torch.no_grad(): pred0,embedding0,hidden0=model(x0.unsqueeze(0).to(device),return_sequence=True)
    embedding=embedding0[0].cpu().numpy().astype(np.float32)
    np.save(output_embeddings/"E_st.npy",embedding)
    metadata={
        "shape":list(embedding.shape),"dtype":str(embedding.dtype),"target_row_index":int(row0),
        "history_row_range_inclusive":[int(row0)-config.sequence_length,int(row0)-1],
        "node_indices":list(range(traffic.shape[1])),
        "node_index_semantics":"Matrix column index only; not a geographic FID and not a coordinate.",
        "timestamp_status":"Unavailable in the headerless source export; no timestamp is fabricated.",
        "checkpoint":str(checkpoint_path.relative_to(root)),"standardizer":scaler.as_dict(),
    }
    (output_embeddings/"E_st_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(output_metrics/"st_evaluation_arrays.npz",predictions=pred,targets=target,target_row_indices=target_rows,
                        baseline_predictions=baseline,selected_hidden_sequence=hidden0[0].cpu().numpy(),
                        selected_prediction=scaler.inverse(pred0[0].cpu().numpy()),selected_target=scaler.inverse(y0.numpy()))
    elapsed=time.perf_counter()-started
    report={
        "stage":"Step 4 spatiotemporal graph encoder","config":asdict(config),"source_paths":source_paths,
        "data_shape":{"traffic":list(traffic.shape),"adjacency":list(adjacency.shape)},
        "split_boundaries_row_ranges_half_open":{k:list(v) for k,v in bounds.items()},
        "datasets":{k:v.audit() for k,v in datasets.items()},"standardizer":scaler.as_dict(),
        "forward_shape_test":{"input":[2,config.sequence_length,traffic.shape[1],1],"embedding":list(test_embedding.shape),"prediction":list(test_prediction.shape),"passed":True},
        "training":{"epochs_completed":len(history),"best_epoch":best_epoch,"early_stopped":len(history)<config.max_epochs,"history":history},
        "test_metrics":model_metrics,"persistence_baseline_test_metrics":baseline_metrics,
        "improvement_over_persistence_percent":{"mae":float((baseline_metrics["mae"]-model_metrics["mae"])/baseline_metrics["mae"]*100),"rmse":float((baseline_metrics["rmse"]-model_metrics["rmse"])/baseline_metrics["rmse"]*100)},
        "parameter_count":int(sum(p.numel() for p in model.parameters())),"runtime_seconds":float(elapsed),
        "artifacts":{"checkpoint":str(checkpoint_path.relative_to(root)),"embedding":"outputs/embeddings/E_st.npy","embedding_metadata":"outputs/embeddings/E_st_metadata.json","evaluation_arrays":"outputs/metrics/st_evaluation_arrays.npz"},
        "sampling_strategy":"Full chronological dataset; no temporal or node subsampling during training/evaluation.",
    }
    metrics_path=output_metrics/"st_training_metrics.json"; metrics_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"test_metrics":model_metrics,"baseline":baseline_metrics,"runtime_seconds":elapsed},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--max-epochs",type=int,default=15); parser.add_argument("--patience",type=int,default=4)
    args=parser.parse_args(); run_experiment(config=TrainingConfig(max_epochs=args.max_epochs,patience=args.patience))


if __name__=="__main__": main()

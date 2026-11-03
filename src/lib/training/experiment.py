"""Experimentos del Modelo A: cada corrida se describe con un `Experiment` y se arma con fábricas.

Así la notebook declara la tabla de experimentos como datos y una sola función los entrena o,
si ya terminaron, los reanuda desde `last.pt` y reimprime su log sin volver a entrenar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.models import ArcFaceHead, CustomCNNConfig, CustomFaceCNN, SoftmaxHead
from lib.training.trainer import CheckpointStore, EarlyStopping, Trainer


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    head: Literal["softmax", "arcface"] = "softmax"
    optimizer: Literal["sgd", "adamw"] = "sgd"
    lr: float = 0.1
    weight_decay: float = 5e-4
    schedule: Literal["cosine", "constant"] = "cosine"
    warmup_epochs: int = 2
    epochs: int = 60
    patience: int = 10
    scale: float = 30.0
    margin: float = 0.5
    init_from: str | None = None  # experimento cuyo best.pt inicializa el extractor

    def as_row(self) -> dict:
        return asdict(self)


def build_head(exp: Experiment, embedding_dim: int, num_classes: int) -> nn.Module:
    if exp.head == "arcface":
        return ArcFaceHead(embedding_dim, num_classes, scale=exp.scale, margin=exp.margin)
    return SoftmaxHead(embedding_dim, num_classes)


def build_optimizer(exp: Experiment, params) -> torch.optim.Optimizer:
    if exp.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=exp.lr, weight_decay=exp.weight_decay)
    return torch.optim.SGD(params, lr=exp.lr, momentum=0.9, nesterov=True, weight_decay=exp.weight_decay)


def build_scheduler(exp: Experiment, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """Coseno con warmup lineal (desde lr/10), o lr constante para fine-tuning."""
    if exp.schedule == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=exp.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp.epochs - exp.warmup_epochs),
        ],
        milestones=[exp.warmup_epochs],
    )


def run_experiment(
    exp: Experiment,
    model_config: CustomCNNConfig,
    num_classes: int,
    loaders: dict[str, DataLoader],
    plain_train_loader: DataLoader,
    checkpoints_root: str | Path,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Entrena `exp` o, si ya tiene `last.pt`, lo reanuda (un experimento terminado solo reimprime su log).

    `loaders` necesita "train", "val_closed" y "val_open". `plain_train_loader` es train sin augmentation
    ni shuffle, para inicializar la cabeza ArcFace con centroides al hacer fine-tuning.
    """
    root = Path(checkpoints_root)
    store = CheckpointStore(root / exp.name)
    model = (CustomFaceCNN.from_checkpoint(root / exp.init_from / "best.pt") if exp.init_from
             else CustomFaceCNN(model_config))
    head = build_head(exp, model.embedding_dim, num_classes)
    optimizer = build_optimizer(exp, list(model.parameters()) + list(head.parameters()))
    trainer = Trainer(model, head, optimizer, build_scheduler(exp, optimizer), device, amp=True)

    log(f"===== {exp.name}: {exp.description}")
    if exp.init_from and isinstance(head, ArcFaceHead) and not store.last_path.exists():
        embeddings = torch.from_numpy(trainer.embed(plain_train_loader))
        head.init_from_centroids(embeddings, torch.from_numpy(plain_train_loader.dataset.labels))
    return trainer.fit(
        loaders["train"], loaders["val_closed"], loaders["val_open"],
        epochs=exp.epochs, early_stopping=EarlyStopping(patience=exp.patience), store=store, resume=True, log=log,
    )

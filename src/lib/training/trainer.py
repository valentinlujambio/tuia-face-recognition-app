"""Loop de entrenamiento del Modelo A con AMP, early stopping y checkpoints reanudables.

Responsabilidades separadas: EarlyStopping decide cuándo cortar, CheckpointStore persiste
el estado (pensado para Google Drive, así una sesión de Colab cortada se retoma) y Trainer
solo entrena y mide. La cabeza es intercambiable (lib.models.heads).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lib.training.metrics import verification_score
from lib.vision import l2_normalize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Corta cuando la métrica monitoreada (a maximizar) no mejora en `patience` épocas."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Registra la métrica de la época; devuelve True si es la mejor hasta ahora."""
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return dict(vars(self))

    def load_state_dict(self, state: dict) -> None:
        vars(self).update(state)


class CheckpointStore:
    """`last.pt`: estado completo para reanudar. `best.pt`: pesos del extractor con mejor métrica."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def history_path(self) -> Path:
        return self.directory / "history.json"

    def save_last(self, state: dict) -> None:
        self._atomic_save(state, self.last_path)

    def save_best(self, model: nn.Module, epoch: int, metric: float) -> None:
        config = asdict(model.config) if hasattr(model, "config") else None
        self._atomic_save({"state_dict": model.state_dict(), "config": config, "epoch": epoch, "metric": metric},
                          self.best_path)

    def load_last(self, map_location: str | torch.device = "cpu") -> dict | None:
        return torch.load(self.last_path, map_location=map_location, weights_only=False) if self.last_path.exists() else None

    def save_history(self, history: list[dict]) -> None:
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    @staticmethod
    def _atomic_save(obj: dict, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)


@dataclass
class EpochLog:
    epoch: int
    lr: float
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_open_auc: float
    val_open_eer: float
    seconds: float
    best: bool = field(default=False)

    def __str__(self) -> str:
        mark = " *" if self.best else ""
        return (f"ep {self.epoch:3d} | lr {self.lr:.4f} | train loss {self.train_loss:.4f} acc {self.train_acc:.3f} | "
                f"val loss {self.val_loss:.4f} acc {self.val_acc:.3f} | open AUC {self.val_open_auc:.4f} "
                f"EER {self.val_open_eer:.4f} | {self.seconds:.0f}s{mark}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        amp: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.head = head.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.label_smoothing = label_smoothing

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp)

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        self.head.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.head(self.model(images), labels)
                loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Loss y accuracy de clasificación sobre identidades vistas en train (val_closed)."""
        self.model.eval()
        self.head.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with self._autocast():
                logits = self.head(self.model(images), labels)
            total_loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def embed(self, loader: DataLoader) -> np.ndarray:
        """Embeddings L2 (N, D) float32, en el orden del loader (sin shuffle)."""
        self.model.eval()
        chunks = []
        for images, _ in loader:
            with self._autocast():
                chunks.append(self.model(images.to(self.device)).float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks)).astype(np.float32)

    def state_dict(self, epoch: int, early_stopping: EarlyStopping, history: list[dict]) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "early_stopping": early_stopping.state_dict(),
            "history": history,
        }

    def load_state_dict(self, state: dict, early_stopping: EarlyStopping) -> tuple[int, list[dict]]:
        self.model.load_state_dict(state["model"])
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        early_stopping.load_state_dict(state["early_stopping"])
        return state["epoch"] + 1, state["history"]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        open_loader: DataLoader,
        epochs: int,
        early_stopping: EarlyStopping,
        store: CheckpointStore,
        resume: bool = True,
        log: Callable[[str], None] = print,
    ) -> list[dict]:
        """Entrena hasta `epochs` o early stopping. Elige el mejor checkpoint por AUC de verificación en val_open."""
        start, history = 0, []
        state = store.load_last(self.device) if resume else None
        if state is not None:
            start, history = self.load_state_dict(state, early_stopping)
            log(f"Reanudando desde la época {start} (mejor AUC {early_stopping.best:.4f} en la época {early_stopping.best_epoch})")
            for row in history:
                log(str(EpochLog(**row)))

        open_labels = open_loader.dataset.labels
        for epoch in range(start, epochs):
            if early_stopping.should_stop:
                break
            t0 = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            score = verification_score(self.embed(open_loader), open_labels)
            self.scheduler.step()

            is_best = early_stopping.step(score.auc, epoch)
            if is_best:
                store.save_best(self.model, epoch, score.auc)
            row = EpochLog(epoch, lr, train_loss, train_acc, val_loss, val_acc, score.auc, score.eer,
                           time.perf_counter() - t0, is_best)
            history.append(asdict(row))
            store.save_last(self.state_dict(epoch, early_stopping, history))
            store.save_history(history)
            log(str(row))

        log(f"Fin: mejor AUC val_open {early_stopping.best:.4f} en la época {early_stopping.best_epoch}")
        return history

"""
models/trainer.py
=================
Ciclo de entrenamiento de la PINN-ORC.

Gestiona:
- Entrenamiento por épocas con los tres términos de pérdida
- Validación y Early Stopping
- Registro de métricas y guardado de checkpoints
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class PINNTrainer:
    """
    Entrenador de la PINN-ORC.

    Parámetros
    ----------
    model      : instancia de PINN_ORC
    loss_fn    : instancia de PINNLoss
    normalizer : DataNormalizer ajustado
    config     : dict de configuración de entrenamiento
    device     : 'cpu' o 'cuda'
    """

    def __init__(self, model, loss_fn, normalizer, config: dict = None, device: str = "cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.normalizer = normalizer
        self.device = device

        cfg = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "n_epochs": 200,
            "patience": 25,
            "lr_patience": 10,
            "lr_factor": 0.5,
            "checkpoint_dir": "results/checkpoints",
            "p_geo_bounds": (5.0, 30.0),
            "grad_clip": 1.0,
        }
        if config:
            cfg.update(config)
        self.cfg = cfg

        Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=cfg["lr_patience"],
            factor=cfg["lr_factor"]
        )
        self.early_stop = EarlyStopping(patience=cfg["patience"])
        self.history = {k: [] for k in [
            "train_total", "train_data", "train_physics", "train_opt",
            "val_total", "val_data", "lr"
        ]}

    def _extract_batch(self, batch):
        x_norm = batch["x_norm"].to(self.device)
        y_raw = batch["y_raw"].to(self.device)
        x_raw = batch["x_raw"].to(self.device)

        # Variables físicas en escala original
        T_geo   = x_raw[:, 0]
        m_geo   = x_raw[:, 1]
        P_geo   = x_raw[:, 5]

        # Targets reales
        targets = {
            "W_net_true":    y_raw[:, 0],
            "m_wf_true":     y_raw[:, 1],
            "T_geo_out_true": y_raw[:, 2],
        }
        inputs_phys = {"T_geo": T_geo, "m_geo": m_geo, "P_geo": P_geo}

        return x_norm, targets, inputs_phys

    def _step(self, batch) -> dict:
        x_norm, targets, inputs_phys = self._extract_batch(batch)
        preds = self.model(x_norm)

        # Desnormalizar W_net para que la escala de L_opt sea física
        W_net_denorm = torch.tensor(
            self.normalizer.Y_range[0] if hasattr(self.normalizer, "Y_range") else 1.0,
            device=self.device
        ) * preds["W_net"]

        preds_denorm = {**preds, "W_net": W_net_denorm}

        loss_dict = self.loss_fn(
            predictions=preds_denorm,
            targets=targets,
            inputs_physical=inputs_phys,
            P_geo_bounds=self.cfg["p_geo_bounds"],
        )
        return loss_dict

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        totals = {"total": 0.0, "data": 0.0, "physics": 0.0, "opt": 0.0}
        n = 0

        for batch in train_loader:
            self.optimizer.zero_grad()
            loss_dict = self._step(batch)
            loss = loss_dict["total"]
            loss.backward()

            if self.cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])

            self.optimizer.step()

            bs = batch["x_norm"].size(0)
            totals["total"]   += loss.item() * bs
            totals["data"]    += loss_dict["data"]["total"].item() * bs
            totals["physics"] += loss_dict["physics"]["total"].item() * bs
            totals["opt"]     += loss_dict["optimization"]["total"].item() * bs
            n += bs

        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        self.model.eval()
        totals = {"total": 0.0, "data": 0.0}
        n = 0

        for batch in val_loader:
            loss_dict = self._step(batch)
            bs = batch["x_norm"].size(0)
            totals["total"] += loss_dict["total"].item() * bs
            totals["data"]  += loss_dict["data"]["total"].item() * bs
            n += bs

        return {k: v / n for k, v in totals.items()}

    def train(self, train_loader, val_loader) -> dict:
        print(f"\n{'='*60}")
        print(f"  Entrenamiento PINN-ORC  |  {self.model.count_parameters():,} parámetros")
        print(f"  Dispositivo: {self.device}  |  Épocas máx: {self.cfg['n_epochs']}")
        print(f"{'='*60}")

        best_val = float("inf")
        best_epoch = 0

        for epoch in range(1, self.cfg["n_epochs"] + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m   = self.validate(val_loader)
            elapsed = time.time() - t0

            # Registrar historia
            self.history["train_total"].append(train_m["total"])
            self.history["train_data"].append(train_m["data"])
            self.history["train_physics"].append(train_m["physics"])
            self.history["train_opt"].append(train_m["opt"])
            self.history["val_total"].append(val_m["total"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            self.scheduler.step(val_m["total"])

            # Guardar mejor modelo
            if val_m["total"] < best_val:
                best_val = val_m["total"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": best_val,
                    },
                    f"{self.cfg['checkpoint_dir']}/best_model.pt",
                )

            # Log
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Época {epoch:4d}/{self.cfg['n_epochs']} | "
                    f"L_train={train_m['total']:.4f} "
                    f"(data={train_m['data']:.4f} "
                    f"phy={train_m['physics']:.4f} "
                    f"opt={train_m['opt']:.4f}) | "
                    f"L_val={val_m['total']:.4f} | "
                    f"t={elapsed:.1f}s"
                )

            if self.early_stop.step(val_m["total"]):
                print(f"\nEarly stopping en época {epoch}. Mejor época: {best_epoch}")
                break

        print(f"\nEntrenamiento completado. Mejor val_loss={best_val:.5f} (época {best_epoch})")
        return self.history

    def load_best(self):
        ckpt = torch.load(f"{self.cfg['checkpoint_dir']}/best_model.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Mejor modelo cargado (época {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")

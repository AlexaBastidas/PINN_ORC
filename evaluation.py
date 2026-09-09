"""
utils/evaluation.py
====================
Funciones de evaluación y visualización de resultados de la PINN-ORC.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def evaluate_model(model, test_loader, normalizer, device="cpu") -> dict:
    """
    Evalúa el modelo en el conjunto de test.

    Retorna métricas: MAE, RMSE, R² para W_net, m_wf, T_geo_out.
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_norm = batch["x_norm"].to(device)
            y_raw  = batch["y_raw"].to(device)
            out    = model(x_norm)

            # Desnormalizar W_net
            W_scale = torch.tensor(
                normalizer.Y_range[0] if hasattr(normalizer, "Y_range") else 1.0,
                device=device
            )
            W_net_phys = out["W_net"] * W_scale

            pred_batch = torch.stack([W_net_phys, out["m_wf"], out["T_geo_out"]], dim=1)
            preds.append(pred_batch.cpu().numpy())
            trues.append(y_raw.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    names = ["W_net [kW]", "m_wf [kg/s]", "T_geo_out [°C]"]
    metrics = {}
    for i, name in enumerate(names):
        p, t = preds[:, i], trues[:, i]
        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  {name:20s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

    return {"metrics": metrics, "predictions": preds, "targets": trues, "names": names}


def plot_training_history(history: dict, save_path: str = "results/training_history.png"):
    """Grafica la evolución de las pérdidas durante el entrenamiento."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_total"]) + 1)

    # Pérdida total
    axes[0].semilogy(epochs, history["train_total"], label="Train", color="#2196F3")
    axes[0].semilogy(epochs, history["val_total"],   label="Val",   color="#FF5722")
    axes[0].set_title("Pérdida Total", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("L_total (log)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Componentes de pérdida (entrenamiento)
    axes[1].semilogy(epochs, history["train_data"],    label="L_data",    color="#4CAF50")
    axes[1].semilogy(epochs, history["train_physics"], label="L_physics", color="#9C27B0")
    axes[1].semilogy(epochs, history["train_opt"],     label="L_opt",     color="#FF9800")
    axes[1].set_title("Componentes de Pérdida", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Pérdida (log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].semilogy(epochs, history["lr"], color="#607D8B")
    axes[2].set_title("Learning Rate", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("LR (log)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Historia de entrenamiento guardada: {save_path}")


def plot_parity(eval_results: dict, save_path: str = "results/parity_plots.png"):
    """Gráficas de paridad: predicción vs. valor real."""
    preds  = eval_results["predictions"]
    trues  = eval_results["targets"]
    names  = eval_results["names"]
    n_vars = len(names)

    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (ax, name) in enumerate(zip(axes, names)):
        p, t = preds[:, i], trues[:, i]
        ax.scatter(t, p, alpha=0.4, s=10, color=colors[i % len(colors)])
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Ideal")
        r2 = eval_results["metrics"][name]["R2"]
        ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulador ORC")
        ax.set_ylabel("PINN")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráficas de paridad guardadas: {save_path}")


def plot_optimization_curve(
    model,
    normalizer,
    T_geo: float = 150.0,
    m_geo: float = 20.0,
    T_amb: float = 25.0,
    f_wf: list = None,
    device: str = "cpu",
    save_path: str = "results/optimization_curve.png",
):
    """
    Grafica W_net vs P_geo para encontrar visualmente la presión óptima.
    """
    if f_wf is None:
        f_wf = [154.05, 36.40]  # R245fa por defecto

    result = model.predict_optimal_pressure(
        T_geo=T_geo, m_geo=m_geo, T_amb=T_amb, f_wf=f_wf,
        normalizer=normalizer, n_points=300, device=device
    )

    grid   = result["grid"]
    P_vals = [r["P_geo"]  for r in grid]
    W_vals = [r["W_net"]  for r in grid]
    P_opt  = result["optimal"]["P_geo"]
    W_opt  = result["optimal"]["W_net"]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(P_vals, W_vals, color="#2196F3", lw=2, label="W_net (PINN)")
    ax.axvline(P_opt, color="#FF5722", ls="--", lw=1.5,
               label=f"P_opt = {P_opt:.1f} bar")
    ax.scatter([P_opt], [W_opt], color="#FF5722", s=80, zorder=5)
    ax.annotate(
        f"W_net_opt = {W_opt:.1f} kW",
        xy=(P_opt, W_opt), xytext=(P_opt + 1.5, W_opt * 0.97),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )
    ax.set_xlabel("Presión geotérmica P_geo [bar]", fontsize=12)
    ax.set_ylabel("Potencia neta W_net [kW]", fontsize=12)
    ax.set_title(
        f"Curva de optimización PINN\n"
        f"T_geo={T_geo}°C  m_geo={m_geo} kg/s  T_amb={T_amb}°C",
        fontsize=12, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curva de optimización guardada: {save_path}")
    print(f"  → P_opt = {P_opt:.2f} bar  |  W_net_opt = {W_opt:.2f} kW")
    return result

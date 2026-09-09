"""
main.py
=======
Script principal de la PINN-ORC.

Flujo:
  1. Generar dataset sintético del simulador ORC
  2. Preparar DataLoaders (train / val / test)
  3. Instanciar la PINN y la función de pérdida
  4. Entrenar con el ciclo PINN
  5. Evaluar en test y generar gráficas
  6. Ejecutar optimización de presión para condiciones de ejemplo
"""

import sys
import os

# Asegurar que las carpetas del proyecto estén en el path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from data.orc_data_generator import (
    generate_dataset,
    DataNormalizer,
    prepare_dataloaders,
)
from models.pinn_orc import PINN_ORC
from models.loss_functions import PINNLoss
from models.trainer import PINNTrainer
from utils.evaluation import (
    evaluate_model,
    plot_training_history,
    plot_parity,
    plot_optimization_curve,
)


# ---------------------------------------------------------------------------
# Configuración del experimento
# ---------------------------------------------------------------------------

CONFIG = {
    # Dataset
    "n_samples": 3000,
    "T_geo_range": (100.0, 180.0),   # °C
    "m_geo_range": (5.0, 50.0),      # kg/s
    "T_amb_range": (15.0, 35.0),     # °C
    "P_geo_range": (5.0, 30.0),      # bar
    "seed": 42,

    # Arquitectura PINN
    "n_inputs": 6,
    "n_hidden": 64,
    "n_layers": 4,
    "activation": "tanh",
    "p_geo_bounds": (5.0, 30.0),

    # Pesos de la función de pérdida
    "lambda_data": 1.0,
    "lambda_physics": 0.5,
    "lambda_opt": 0.1,

    # Entrenamiento
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "n_epochs": 200,
    "batch_size": 64,
    "patience": 25,
    "lr_patience": 10,
    "lr_factor": 0.5,
    "grad_clip": 1.0,

    # Rutas
    "checkpoint_dir": "results/checkpoints",

    # Dispositivo
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    print("\n" + "=" * 60)
    print("  PINN-ORC — Optimización embebida en entrenamiento")
    print("  Ciclo Rankine Orgánico + Recurso Geotérmico")
    print("=" * 60)
    print(f"  Dispositivo: {CONFIG['device']}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ------------------------------------------------------------------
    # 1. Generar dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Generando dataset sintético...")
    dataset = generate_dataset(
        n_samples=CONFIG["n_samples"],
        T_geo_range=CONFIG["T_geo_range"],
        m_geo_range=CONFIG["m_geo_range"],
        T_amb_range=CONFIG["T_amb_range"],
        P_geo_range=CONFIG["P_geo_range"],
        seed=CONFIG["seed"],
    )

    # ------------------------------------------------------------------
    # 2. Normalizar y preparar DataLoaders
    # ------------------------------------------------------------------
    print("\n[2/5] Preparando DataLoaders...")
    normalizer = DataNormalizer(method="minmax")
    train_loader, val_loader, test_loader, normalizer = prepare_dataloaders(
        dataset_dict=dataset,
        normalizer=normalizer,
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )

    # ------------------------------------------------------------------
    # 3. Instanciar modelo y función de pérdida
    # ------------------------------------------------------------------
    print("\n[3/5] Construyendo PINN...")
    model = PINN_ORC(
        n_inputs=CONFIG["n_inputs"],
        n_hidden=CONFIG["n_hidden"],
        n_layers=CONFIG["n_layers"],
        activation=CONFIG["activation"],
        p_geo_bounds=CONFIG["p_geo_bounds"],
    )
    print(f"  Parámetros entrenables: {model.count_parameters():,}")

    loss_fn = PINNLoss(
        lambda_data=CONFIG["lambda_data"],
        lambda_physics=CONFIG["lambda_physics"],
        lambda_opt=CONFIG["lambda_opt"],
    )

    # ------------------------------------------------------------------
    # 4. Entrenar
    # ------------------------------------------------------------------
    print("\n[4/5] Iniciando entrenamiento...")
    trainer = PINNTrainer(
        model=model,
        loss_fn=loss_fn,
        normalizer=normalizer,
        config=CONFIG,
        device=CONFIG["device"],
    )
    history = trainer.train(train_loader, val_loader)
    trainer.load_best()

    plot_training_history(history)

    # ------------------------------------------------------------------
    # 5. Evaluar en test
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluando en conjunto de test...")
    eval_results = evaluate_model(model, test_loader, normalizer, device=CONFIG["device"])
    plot_parity(eval_results)

    # ------------------------------------------------------------------
    # 6. Optimización de presión para condición de ejemplo
    # ------------------------------------------------------------------
    print("\n[Optimización] Buscando P_geo óptima para condición de ejemplo...")
    opt_result = plot_optimization_curve(
        model=model,
        normalizer=normalizer,
        T_geo=150.0,
        m_geo=20.0,
        T_amb=25.0,
        f_wf=[154.05, 36.40],   # R245fa
        device=CONFIG["device"],
    )

    print("\n" + "=" * 60)
    print("  Ejecución completada exitosamente.")
    print(f"  Resultados en: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

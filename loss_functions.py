"""
models/loss_functions.py
========================
Función de pérdida total de la PINN-ORC:

    L_total = λ1 * L_data + λ2 * L_physics + λ3 * L_opt

donde:
    L_data    : error cuadrático medio vs. datos del simulador ORC
    L_physics : incumplimiento de ecuaciones físicas y restricciones
    L_opt     : -W_net (negativo para maximizar mediante minimización)
"""

import torch
import torch.nn as nn
from physics.orc_physics import ORCPhysics


class PINNLoss(nn.Module):
    """
    Función de pérdida compuesta para la PINN-ORC.

    Parámetros
    ----------
    lambda_data    : peso del término de datos (λ1)
    lambda_physics : peso del término de física (λ2)
    lambda_opt     : peso del término de optimización (λ3)
    physics_config : configuración de parámetros físicos del ORC
    reduction      : 'mean' o 'sum' para reducción de MSE
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.5,
        lambda_opt: float = 0.1,
        physics_config: dict = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_opt = lambda_opt
        self.reduction = reduction

        # Módulo de física
        self.physics = ORCPhysics(config=physics_config)

    # ------------------------------------------------------------------
    # Término 1: Pérdida de datos
    # ------------------------------------------------------------------

    def data_loss(
        self,
        W_net_pred: torch.Tensor,
        W_net_true: torch.Tensor,
        m_wf_pred: torch.Tensor = None,
        m_wf_true: torch.Tensor = None,
        T_geo_out_pred: torch.Tensor = None,
        T_geo_out_true: torch.Tensor = None,
        weights: dict = None,
    ) -> dict:
        """
        Error cuadrático medio entre predicciones y datos del simulador.

        Puede incluir múltiples salidas (W_net, m_wf, T_geo_out) con
        pesos individuales.

        Parámetros
        ----------
        weights : dict con pesos para cada variable, p.ej.:
                  {'W_net': 1.0, 'm_wf': 0.3, 'T_geo_out': 0.2}
        """
        if weights is None:
            weights = {"W_net": 1.0, "m_wf": 0.3, "T_geo_out": 0.2}

        losses = {}

        # Error en potencia neta (principal)
        mse_wnet = nn.functional.mse_loss(W_net_pred, W_net_true, reduction=self.reduction)
        losses["W_net"] = mse_wnet * weights.get("W_net", 1.0)

        # Error en caudal del fluido de trabajo (si disponible)
        if m_wf_pred is not None and m_wf_true is not None:
            mse_mwf = nn.functional.mse_loss(m_wf_pred, m_wf_true, reduction=self.reduction)
            losses["m_wf"] = mse_mwf * weights.get("m_wf", 0.3)

        # Error en temperatura de salida geotérmica (si disponible)
        if T_geo_out_pred is not None and T_geo_out_true is not None:
            mse_tgeo = nn.functional.mse_loss(
                T_geo_out_pred, T_geo_out_true, reduction=self.reduction
            )
            losses["T_geo_out"] = mse_tgeo * weights.get("T_geo_out", 0.2)

        losses["total"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Término 2: Pérdida de física
    # ------------------------------------------------------------------

    def physics_loss(
        self,
        T_geo: torch.Tensor,
        m_geo: torch.Tensor,
        P_geo: torch.Tensor,
        m_wf: torch.Tensor,
        T_geo_out: torch.Tensor,
        W_net: torch.Tensor,
    ) -> dict:
        """
        Residuos de las ecuaciones de conservación y restricciones
        termodinámicas del ORC.
        """
        return self.physics.physics_loss(
            T_geo=T_geo,
            m_geo=m_geo,
            P_geo=P_geo,
            m_wf=m_wf,
            T_geo_out=T_geo_out,
            W_net=W_net,
        )

    # ------------------------------------------------------------------
    # Término 3: Pérdida de optimización
    # ------------------------------------------------------------------

    def optimization_loss(
        self,
        W_net_pred: torch.Tensor,
        P_geo: torch.Tensor,
        P_min: float,
        P_max: float,
    ) -> dict:
        """
        Término que convierte el entrenamiento en optimización:
            L_opt = -W_net  →  minimizar L_opt = maximizar W_net

        Incluye también una penalización suave por violación de los
        límites de presión.

        Parámetros
        ----------
        W_net_pred : potencia neta predicha [kW]
        P_geo      : presiones en el batch [bar]
        P_min, P_max: límites operativos de presión [bar]
        """
        # Maximizar potencia: minimizar negativo de W_net
        L_max_power = -W_net_pred.mean()

        # Penalización suave de límites de presión (barrera logarítmica)
        eps = 1e-3
        P_clamped = torch.clamp(P_geo, min=P_min + eps, max=P_max - eps)
        barrier = -torch.log((P_clamped - P_min) / (P_max - P_min) + eps).mean()
        barrier += -torch.log((P_max - P_clamped) / (P_max - P_min) + eps).mean()

        losses = {
            "maximize_power": L_max_power,
            "pressure_bounds": barrier * 0.01,  # peso pequeño para la barrera
        }
        losses["total"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Pérdida total
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: dict,
        targets: dict,
        inputs_physical: dict,
        P_geo_bounds: tuple = (5.0, 30.0),
    ) -> dict:
        """
        Calcula la pérdida total de la PINN.

        Parámetros
        ----------
        predictions : salidas de la red neuronal
            {'W_net', 'm_wf', 'T_geo_out'}

        targets : valores del simulador ORC (etiquetas)
            {'W_net_true', 'm_wf_true', 'T_geo_out_true'} (opcionales algunos)

        inputs_physical : variables físicas en escala original
            {'T_geo', 'm_geo', 'P_geo', 'T_amb'}

        P_geo_bounds : (P_min, P_max) para la penalización de límites

        Retorna
        -------
        dict con todas las componentes de pérdida y la pérdida total.
        """
        P_min, P_max = P_geo_bounds

        # --- L_data ---
        l_data = self.data_loss(
            W_net_pred=predictions["W_net"],
            W_net_true=targets["W_net_true"],
            m_wf_pred=predictions.get("m_wf"),
            m_wf_true=targets.get("m_wf_true"),
            T_geo_out_pred=predictions.get("T_geo_out"),
            T_geo_out_true=targets.get("T_geo_out_true"),
        )

        # --- L_physics ---
        l_phys = self.physics_loss(
            T_geo=inputs_physical["T_geo"],
            m_geo=inputs_physical["m_geo"],
            P_geo=inputs_physical["P_geo"],
            m_wf=predictions["m_wf"],
            T_geo_out=predictions["T_geo_out"],
            W_net=predictions["W_net"],
        )

        # --- L_opt ---
        l_opt = self.optimization_loss(
            W_net_pred=predictions["W_net"],
            P_geo=inputs_physical["P_geo"],
            P_min=P_min,
            P_max=P_max,
        )

        # --- Total ponderado ---
        total = (
            self.lambda_data * l_data["total"]
            + self.lambda_physics * l_phys["total"]
            + self.lambda_opt * l_opt["total"]
        )

        return {
            "total": total,
            "data": l_data,
            "physics": l_phys,
            "optimization": l_opt,
            "lambdas": {
                "data": self.lambda_data,
                "physics": self.lambda_physics,
                "opt": self.lambda_opt,
            },
        }

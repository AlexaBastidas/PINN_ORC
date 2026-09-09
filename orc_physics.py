"""
physics/orc_physics.py
======================
Ecuaciones físicas del Ciclo Rankine Orgánico (ORC) geotérmico.

Implementa los balances de masa, energía y la restricción de potencia
neta positiva descritos en la sección "Formulación física del Sistema
ORC para la implementación en PINNs". Las propiedades termodinámicas
del fluido de trabajo (entalpía, entropía, temperatura de saturación,
densidad) se obtienen mediante la librería CoolProp [Bell2014], que
implementa ecuaciones de estado equivalentes a REFPROP del NIST.

Como CoolProp no es diferenciable mediante autograd de PyTorch, las
propiedades se precalculan sobre una grilla de presiones y se exponen
a la red mediante interpolación lineal diferenciable (torch.searchsorted
+ combinación convexa), preservando la propagación de gradientes
necesaria para entrenar la PINN.

Notación:
    - geo  : fluido geotérmico (fuente de calor)
    - wf   : fluido de trabajo del ORC
    - cond : condensador
    - evap : evaporador
    - exp  : expansor (turbina)
    - pump : bomba

Estados termodinámicos del fluido de trabajo (convención Zare, 2015;
Quoilin et al., 2013):
    1 : vapor a la entrada del expansor (salida del evaporador)
    2 : vapor/líquido a la salida del expansor (entrada del condensador)
    3 : líquido saturado a la salida del condensador
    4 : líquido comprimido a la salida de la bomba (entrada del evaporador)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from CoolProp.CoolProp import PropsSI
    _COOLPROP_AVAILABLE = True
except ImportError:
    _COOLPROP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parámetros físicos por defecto (pueden sobreescribirse en la configuración)
# ---------------------------------------------------------------------------
DEFAULT_PHYSICS = {
    # Eficiencias isentrópicas — valores típicos para ORC geotérmico
    # (Zare, 2015; Quoilin et al., 2013): eta_exp = 0.75-0.85, eta_gen = 0.95-0.98
    "eta_exp": 0.80,        # expansor (turbina)
    "eta_pump": 0.75,       # bomba
    "eta_gen": 0.95,        # generador eléctrico

    # Calor específico del fluido geotérmico [kJ/(kg·K)] — agua/brine
    "cp_geo": 4.18,

    # Fluido de trabajo (nombre reconocido por CoolProp)
    "fluido_wf": "R245fa",

    # Presión de condensación de referencia [bar]
    "P_cond_ref": 3.0,

    # Rango de presión de evaporación soportado por la tabla de propiedades [bar]
    # (debe mantenerse por debajo de la presión crítica del fluido seleccionado;
    # R245fa: Pcrit ≈ 36.5 bar)
    "P_evap_min": 2.0,
    "P_evap_max": 33.0,
    "n_grid_points": 400,
}


# ---------------------------------------------------------------------------
# Tabla de propiedades termodinámicas (CoolProp) con interpolación diferenciable
# ---------------------------------------------------------------------------
class _PropertyTable:
    """
    Precalcula propiedades de saturación del fluido de trabajo sobre una
    grilla de presiones usando CoolProp, y expone una interpolación lineal
    diferenciable respecto a la presión para uso dentro de la PINN.

    Esta clase resuelve la incompatibilidad entre CoolProp (cómputo en C++,
    no diferenciable) y PyTorch autograd: la grilla se calcula una sola vez
    en __init__, y las consultas durante el entrenamiento son operaciones
    tensoriales (búsqueda + interpolación) que sí propagan gradiente
    respecto a la presión de entrada.
    """

    def __init__(self, fluido: str, p_min_bar: float, p_max_bar: float, n_points: int):
        if not _COOLPROP_AVAILABLE:
            raise ImportError(
                "CoolProp no está instalado. Ejecute: pip install CoolProp"
            )

        self.fluido = fluido
        p_grid_bar = np.linspace(p_min_bar, p_max_bar, n_points)
        p_grid_pa = p_grid_bar * 1e5

        T_sat = np.empty(n_points)   # [°C]
        h_g = np.empty(n_points)     # [kJ/kg] vapor saturado
        h_f = np.empty(n_points)     # [kJ/kg] líquido saturado
        s_g = np.empty(n_points)     # [kJ/(kg·K)] vapor saturado
        v_f = np.empty(n_points)     # [m³/kg] líquido saturado

        for i, p_pa in enumerate(p_grid_pa):
            T_sat[i] = PropsSI("T", "P", p_pa, "Q", 1, fluido) - 273.15
            h_g[i] = PropsSI("H", "P", p_pa, "Q", 1, fluido) / 1000.0
            h_f[i] = PropsSI("H", "P", p_pa, "Q", 0, fluido) / 1000.0
            s_g[i] = PropsSI("S", "P", p_pa, "Q", 1, fluido) / 1000.0
            v_f[i] = 1.0 / PropsSI("D", "P", p_pa, "Q", 0, fluido)

        self.p_grid = torch.tensor(p_grid_bar, dtype=torch.float32)
        self.T_sat = torch.tensor(T_sat, dtype=torch.float32)
        self.h_g = torch.tensor(h_g, dtype=torch.float32)
        self.h_f = torch.tensor(h_f, dtype=torch.float32)
        self.s_g = torch.tensor(s_g, dtype=torch.float32)
        self.v_f = torch.tensor(v_f, dtype=torch.float32)

    def to(self, device):
        self.p_grid = self.p_grid.to(device)
        self.T_sat = self.T_sat.to(device)
        self.h_g = self.h_g.to(device)
        self.h_f = self.h_f.to(device)
        self.s_g = self.s_g.to(device)
        self.v_f = self.v_f.to(device)
        return self

    def _interp(self, table: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Interpolación lineal diferenciable de `table` (definida sobre
        self.p_grid) en los puntos de presión `P` [bar].
        El gradiente respecto a P se propaga correctamente.
        """
        P_clamped = torch.clamp(P, self.p_grid[0], self.p_grid[-1])
        idx = torch.searchsorted(self.p_grid, P_clamped.detach(), right=False)
        idx = torch.clamp(idx, 1, len(self.p_grid) - 1)

        p_lo = self.p_grid[idx - 1]
        p_hi = self.p_grid[idx]
        v_lo = table[idx - 1]
        v_hi = table[idx]

        w = (P_clamped - p_lo) / (p_hi - p_lo + 1e-12)
        return v_lo + w * (v_hi - v_lo)

    def T_sat_of(self, P: torch.Tensor) -> torch.Tensor:
        """Temperatura de saturación T_sat(P) [°C]."""
        return self._interp(self.T_sat, P)

    def h_g_of(self, P: torch.Tensor) -> torch.Tensor:
        """Entalpía de vapor saturado h_g(P) [kJ/kg]."""
        return self._interp(self.h_g, P)

    def h_f_of(self, P: torch.Tensor) -> torch.Tensor:
        """Entalpía de líquido saturado h_f(P) [kJ/kg]."""
        return self._interp(self.h_f, P)

    def s_g_of(self, P: torch.Tensor) -> torch.Tensor:
        """Entropía de vapor saturado s_g(P) [kJ/(kg·K)]."""
        return self._interp(self.s_g, P)

    def v_f_of(self, P: torch.Tensor) -> torch.Tensor:
        """Volumen específico de líquido saturado v_f(P) [m³/kg]."""
        return self._interp(self.v_f, P)


class ORCPhysics(nn.Module):
    """
    Módulo de física del ORC.

    Implementa:
        - Balance de masa (caudal constante en el circuito cerrado / abierto)
        - Balance de energía por componente (evaporador, expansor, condensador, bomba)
        - Balance global de potencia neta (ecuación de W_net)
        - Restricción de potencia neta positiva (penalización ReLU)

    Las propiedades termodinámicas del fluido de trabajo se obtienen de
    CoolProp mediante la tabla diferenciable `_PropertyTable`.

    Parámetros
    ----------
    config : dict
        Parámetros físicos (usa DEFAULT_PHYSICS si no se especifican).
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = DEFAULT_PHYSICS.copy()
        if config is not None:
            cfg.update(config)

        self.fluido_wf = cfg["fluido_wf"]

        # Registrar parámetros escalares como buffers (no entrenables)
        for key in ("eta_exp", "eta_pump", "eta_gen", "cp_geo", "P_cond_ref"):
            self.register_buffer(key, torch.tensor(float(cfg[key])))

        # Tabla de propiedades termodinámicas (CoolProp precalculado)
        self._props = _PropertyTable(
            fluido=cfg["fluido_wf"],
            p_min_bar=cfg["P_evap_min"],
            p_max_bar=cfg["P_evap_max"],
            n_points=cfg["n_grid_points"],
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        module._props = module._props.to(self.eta_exp.device)
        return module

    # ------------------------------------------------------------------
    # Propiedades termodinámicas del fluido de trabajo (vía CoolProp)
    # ------------------------------------------------------------------

    def T_sat_wf(self, P: torch.Tensor) -> torch.Tensor:
        """Temperatura de saturación del fluido de trabajo a presión P [°C]."""
        return self._props.T_sat_of(P)

    def h1_vapor_sat(self, P_evap: torch.Tensor) -> torch.Tensor:
        """Entalpía del estado 1: vapor saturado a P_evap [kJ/kg]."""
        return self._props.h_g_of(P_evap)

    def s1_vapor_sat(self, P_evap: torch.Tensor) -> torch.Tensor:
        """Entropía del estado 1: vapor saturado a P_evap [kJ/(kg·K)]."""
        return self._props.s_g_of(P_evap)

    def h3_liquid_sat(self, P_cond: torch.Tensor) -> torch.Tensor:
        """Entalpía del estado 3: líquido saturado a P_cond [kJ/kg]."""
        return self._props.h_f_of(P_cond)

    def v3_liquid_sat(self, P_cond: torch.Tensor) -> torch.Tensor:
        """Volumen específico del estado 3: líquido saturado a P_cond [m³/kg]."""
        return self._props.v_f_of(P_cond)

    def h2s_isentropic(self, P_evap: torch.Tensor, P_cond: torch.Tensor) -> torch.Tensor:
        """
        Entalpía isentrópica de salida del expansor (estado 2s):
            h_2s = PropsSI('H', 'P', P_cond, 'S', s_1, fluido)

        s_1 se obtiene de la tabla diferenciable. La inversión h(P, s) se
        aproxima localmente mediante una expansión de primer orden a lo
        largo de la isoentrópica, usando como referencia la curva de vapor
        saturado h_g(P) y s_g(P) ya tabulada:
            h_2s ≈ h_g(P_cond) + T_cond · (s_1 - s_g(P_cond))
        """
        s1 = self.s1_vapor_sat(P_evap)
        h_g_cond = self._props.h_g_of(P_cond)
        s_g_cond = self._props.s_g_of(P_cond)
        T_cond = self.T_sat_wf(P_cond) + 273.15  # K
        h2s = h_g_cond + T_cond * (s1 - s_g_cond)
        return h2s

    # ------------------------------------------------------------------
    # Balance de masa
    # ------------------------------------------------------------------

    def mass_balance_residual(self, m_in: torch.Tensor, m_out: torch.Tensor) -> torch.Tensor:
        """
        Residuo del balance de masa en régimen estacionario:
            sum(m_in) - sum(m_out) = 0   →   m_wf constante en el circuito cerrado

        Retorna el residuo (idealmente nulo si m_in == m_out).
        """
        return m_in - m_out

    # ------------------------------------------------------------------
    # Balance de energía — Evaporador
    # ------------------------------------------------------------------

    def Q_evap_geo(
        self, m_geo: torch.Tensor, T_geo_in: torch.Tensor, T_geo_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Calor cedido por el fluido geotérmico:
            Q_evap = m_geo · cp_geo · (T_geo,in - T_geo,out)
        """
        return m_geo * self.cp_geo * (T_geo_in - T_geo_out)

    def Q_evap_wf(self, m_wf: torch.Tensor, h1: torch.Tensor, h4: torch.Tensor) -> torch.Tensor:
        """
        Calor absorbido por el fluido de trabajo:
            Q_evap = m_wf · (h_1 - h_4)
        """
        return m_wf * (h1 - h4)

    def residual_energy_evaporator(
        self,
        m_geo: torch.Tensor,
        T_geo_in: torch.Tensor,
        T_geo_out: torch.Tensor,
        m_wf: torch.Tensor,
        h1: torch.Tensor,
        h4: torch.Tensor,
    ) -> torch.Tensor:
        """
        Residuo del balance global de energía del evaporador:
            m_geo · cp_geo · (T_geo,in - T_geo,out) = m_wf · (h_1 - h_4)

        Retorna el residuo normalizado [-] (cero si el balance se cumple).
        """
        Q_geo = self.Q_evap_geo(m_geo, T_geo_in, T_geo_out)
        Q_wf = self.Q_evap_wf(m_wf, h1, h4)
        return (Q_geo - Q_wf) / (torch.abs(Q_geo) + 1e-6)

    # ------------------------------------------------------------------
    # Balance de energía — Expansor (turbina)
    # ------------------------------------------------------------------

    def h2_real(self, h1: torch.Tensor, h2s: torch.Tensor) -> torch.Tensor:
        """
        Entalpía real a la salida del expansor:
            h_2 = h_1 - eta_exp · (h_1 - h_2s)
        """
        return h1 - self.eta_exp * (h1 - h2s)

    def W_exp(self, m_wf: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Potencia mecánica desarrollada por el expansor:
            W_exp = m_wf · (h_1 - h_2)
        """
        return m_wf * (h1 - h2)

    def W_elec(self, W_exp: torch.Tensor) -> torch.Tensor:
        """
        Potencia eléctrica entregada por el generador:
            W_elec = eta_gen · W_exp
        """
        return self.eta_gen * W_exp

    # ------------------------------------------------------------------
    # Balance de energía — Condensador
    # ------------------------------------------------------------------

    def Q_cond(self, m_wf: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
        """
        Calor rechazado en el condensador:
            Q_cond = m_wf · (h_2 - h_3)
        """
        return m_wf * (h2 - h3)

    # ------------------------------------------------------------------
    # Balance de energía — Bomba
    # ------------------------------------------------------------------

    def w_pump_isentropic(
        self, v3: torch.Tensor, P_evap: torch.Tensor, P_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Trabajo isentrópico específico de la bomba:
            w_pump,s = v_3 · (P_evap - P_cond)

        P_evap, P_cond en bar; v3 en m³/kg. El resultado se expresa en
        kJ/kg aplicando el factor de conversión bar·m³/kg = 100 kJ/kg.
        """
        return v3 * (P_evap - P_cond) * 100.0

    def h4_real(
        self, h3: torch.Tensor, v3: torch.Tensor, P_evap: torch.Tensor, P_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Entalpía real a la salida de la bomba:
            h_4 = h_3 + v_3·(P_evap - P_cond) / eta_pump
        """
        w_s = self.w_pump_isentropic(v3, P_evap, P_cond)
        return h3 + w_s / self.eta_pump

    def W_pump(self, m_wf: torch.Tensor, h4: torch.Tensor, h3: torch.Tensor) -> torch.Tensor:
        """
        Potencia consumida por la bomba:
            W_pump = m_wf · (h_4 - h_3)
        """
        return m_wf * (h4 - h3)

    # ------------------------------------------------------------------
    # Balance global del ciclo — Potencia neta
    # ------------------------------------------------------------------

    def W_net(
        self,
        m_wf: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
        h3: torch.Tensor,
        h4: torch.Tensor,
    ) -> torch.Tensor:
        """
        Potencia neta generada por el ciclo:
            W_net = eta_gen · m_wf · (h_1 - h_2) - m_wf · (h_4 - h_3)
        """
        return self.eta_gen * m_wf * (h1 - h2) - m_wf * (h4 - h3)

    def eta_th(self, W_net: torch.Tensor, Q_evap: torch.Tensor) -> torch.Tensor:
        """
        Eficiencia térmica del ciclo:
            eta_th = W_net / Q_evap
        """
        return W_net / (Q_evap + 1e-6)

    # ------------------------------------------------------------------
    # Restricción de potencia neta positiva
    # ------------------------------------------------------------------

    def constraint_positive_net_power(self, W_net: torch.Tensor) -> torch.Tensor:
        """
        Penalización de la restricción W_net > 0:
            L_Wnet = ReLU(-W_net)^2

        Es nula cuando W_net > 0 y crece cuadráticamente cuando se viola.
        """
        return torch.relu(-W_net) ** 2

    # ------------------------------------------------------------------
    # Evaluación completa del ciclo a partir de variables de operación
    # ------------------------------------------------------------------

    def evaluate_cycle(
        self,
        P_evap: torch.Tensor,
        P_cond: torch.Tensor,
        m_wf: torch.Tensor,
        m_geo: torch.Tensor,
        T_geo_in: torch.Tensor,
        T_geo_out: torch.Tensor,
    ) -> dict:
        """
        Evalúa los cuatro estados termodinámicos y las potencias del ciclo
        ORC completo a partir de las variables de operación.

        Retorna un diccionario con entalpías, potencias y residuos físicos.
        """
        # Estados termodinámicos
        h1 = self.h1_vapor_sat(P_evap)
        h2s = self.h2s_isentropic(P_evap, P_cond)
        h2 = self.h2_real(h1, h2s)
        h3 = self.h3_liquid_sat(P_cond)
        v3 = self.v3_liquid_sat(P_cond)
        h4 = self.h4_real(h3, v3, P_evap, P_cond)

        # Potencias
        W_exp = self.W_exp(m_wf, h1, h2)
        W_elec = self.W_elec(W_exp)
        W_pump = self.W_pump(m_wf, h4, h3)
        W_net = self.W_net(m_wf, h1, h2, h3, h4)

        # Calores
        Q_evap = self.Q_evap_wf(m_wf, h1, h4)
        Q_cond = self.Q_cond(m_wf, h2, h3)

        # Residuo del balance de energía del evaporador
        res_evap = self.residual_energy_evaporator(
            m_geo, T_geo_in, T_geo_out, m_wf, h1, h4
        )

        # Restricción de potencia neta positiva
        L_Wnet = self.constraint_positive_net_power(W_net)

        return {
            "h1": h1, "h2": h2, "h3": h3, "h4": h4,
            "W_exp": W_exp, "W_elec": W_elec, "W_pump": W_pump, "W_net": W_net,
            "Q_evap": Q_evap, "Q_cond": Q_cond,
            "eta_th": self.eta_th(W_net, Q_evap),
            "residual_evaporator": res_evap,
            "L_Wnet": L_Wnet,
        }

    # ------------------------------------------------------------------
    # Pérdida de física total (para uso dentro de PINNLoss)
    # ------------------------------------------------------------------

    def physics_loss(
        self,
        P_evap: torch.Tensor,
        P_cond: torch.Tensor,
        m_wf: torch.Tensor,
        m_geo: torch.Tensor,
        T_geo_in: torch.Tensor,
        T_geo_out: torch.Tensor,
        W_net: torch.Tensor,
    ) -> dict:
        """
        Calcula el residuo del balance de energía del evaporador y la
        penalización de la restricción de potencia neta positiva.

        Retorna un diccionario con cada componente de la pérdida física.
        """
        h1 = self.h1_vapor_sat(P_evap)
        h4 = self.h4_real(
            self.h3_liquid_sat(P_cond), self.v3_liquid_sat(P_cond), P_evap, P_cond
        )
        res_evap = self.residual_energy_evaporator(
            m_geo, T_geo_in, T_geo_out, m_wf, h1, h4
        )
        L_Wnet = self.constraint_positive_net_power(W_net)

        losses = {
            "residual_evaporator": (res_evap ** 2).mean(),
            "violation_net_power": L_Wnet.mean(),
        }
        losses["total"] = sum(losses.values())
        return losses


# ---------------------------------------------------------------------------
# Ejemplos de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Ejemplo 1: instanciar el módulo de física con R245fa ────────────────
    physics = ORCPhysics(config={"fluido_wf": "R245fa"})

    # ── Ejemplo 2: evaluar el ciclo completo para una condición operativa ──
    # Condiciones de operación: T_geo entrada/salida, caudal geotérmico,
    # presión de evaporación y de condensación, y caudal de fluido de trabajo
    P_evap = torch.tensor([15.0])      # bar
    P_cond = torch.tensor([3.0])       # bar
    m_wf = torch.tensor([3.5])         # kg/s
    m_geo = torch.tensor([20.0])       # kg/s
    T_geo_in = torch.tensor([150.0])   # °C
    T_geo_out = torch.tensor([95.0])   # °C

    resultado = physics.evaluate_cycle(
        P_evap=P_evap, P_cond=P_cond, m_wf=m_wf,
        m_geo=m_geo, T_geo_in=T_geo_in, T_geo_out=T_geo_out,
    )

    print("── Ejemplo 1: evaluación completa del ciclo ──")
    print(f"  h1 (vapor sat. evap.) = {resultado['h1'].item():.2f} kJ/kg")
    print(f"  h2 (salida expansor)  = {resultado['h2'].item():.2f} kJ/kg")
    print(f"  h3 (líq. sat. cond.)  = {resultado['h3'].item():.2f} kJ/kg")
    print(f"  h4 (salida bomba)     = {resultado['h4'].item():.2f} kJ/kg")
    print(f"  W_exp  = {resultado['W_exp'].item():.2f} kW")
    print(f"  W_pump = {resultado['W_pump'].item():.2f} kW")
    print(f"  W_net  = {resultado['W_net'].item():.2f} kW")
    print(f"  eta_th = {resultado['eta_th'].item():.4f}")
    print(f"  residuo balance evaporador = {resultado['residual_evaporator'].item():.6f}")
    print(f"  L_Wnet (penalización)      = {resultado['L_Wnet'].item():.6f}")

    # ── Ejemplo 3: cálculo directo de physics_loss para un batch de entrenamiento ──
    print("\n── Ejemplo 2: physics_loss para un batch ──")
    batch_size = 4
    P_evap_batch = torch.tensor([10.0, 15.0, 20.0, 25.0])
    P_cond_batch = torch.full((batch_size,), 3.0)
    m_wf_batch = torch.tensor([2.8, 3.5, 4.1, 4.6])
    m_geo_batch = torch.full((batch_size,), 20.0)
    T_geo_in_batch = torch.full((batch_size,), 150.0)
    T_geo_out_batch = torch.tensor([100.0, 95.0, 90.0, 85.0])

    # Potencia neta predicha por la red (simulada aquí con valores de ejemplo)
    W_net_pred = torch.tensor([55.0, 70.0, 80.0, 60.0], requires_grad=True)

    perdidas = physics.physics_loss(
        P_evap=P_evap_batch, P_cond=P_cond_batch, m_wf=m_wf_batch,
        m_geo=m_geo_batch, T_geo_in=T_geo_in_batch, T_geo_out=T_geo_out_batch,
        W_net=W_net_pred,
    )
    print(f"  residual_evaporator (MSE) = {perdidas['residual_evaporator'].item():.6f}")
    print(f"  violation_net_power       = {perdidas['violation_net_power'].item():.6f}")
    print(f"  L_physics total           = {perdidas['total'].item():.6f}")

    # Verificación de que el gradiente se propaga correctamente
    perdidas["total"].backward()
    print(f"  grad respecto a W_net_pred: {W_net_pred.grad}")

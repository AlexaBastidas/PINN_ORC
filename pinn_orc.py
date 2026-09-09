"""
models/pinn_orc.py
==================
Arquitectura de la Red Neuronal Informada por la Física (PINN)
para optimización del ciclo ORC geotérmico.

La red aproxima:
    W_net = f(T_geo, m_geo, T_amb, f_wf, P_geo)

Adicionalmente, predice variables internas del ciclo necesarias
para evaluar los residuos físicos:
    - m_wf    : caudal másico del fluido de trabajo [kg/s]
    - T_geo_out: temperatura de salida del fluido geotérmico [°C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinActivation(nn.Module):
    """Activación sinusoidal (útil para PINNs con comportamiento periódico)."""
    def forward(self, x):
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    """
    Activación adaptativa: a * tanh(b * x)
    donde a y b son parámetros entrenables.
    Permite que la red ajuste la escala de la activación.
    """
    def __init__(self, n_neurons: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_neurons))
        self.b = nn.Parameter(torch.ones(n_neurons))

    def forward(self, x):
        return self.a * torch.tanh(self.b * x)


class ResidualBlock(nn.Module):
    """
    Bloque residual para facilitar el entrenamiento de redes profundas.
    Implementa: out = activation(W2 * activation(W1 * x + b1) + b2) + x
    """
    def __init__(self, n_neurons: int, activation: str = "tanh"):
        super().__init__()
        self.linear1 = nn.Linear(n_neurons, n_neurons)
        self.linear2 = nn.Linear(n_neurons, n_neurons)
        self._activation = self._get_activation(activation, n_neurons)

    def _get_activation(self, name: str, n: int):
        if name == "tanh":
            return nn.Tanh()
        elif name == "sin":
            return SinActivation()
        elif name == "swish":
            return nn.SiLU()
        elif name == "adaptive":
            return AdaptiveActivation(n)
        else:
            return nn.Tanh()

    def forward(self, x):
        h = self._activation(self.linear1(x))
        h = self.linear2(h)
        return self._activation(h + x)


class PINN_ORC(nn.Module):
    """
    Red Neuronal Informada por la Física para el ORC geotérmico.

    Arquitectura
    ------------
    - Capa de entrada: proyección al espacio latente
    - Bloques residuales: capas ocultas con conexiones residuales
    - Cabezas de salida múltiples: W_net, m_wf, T_geo_out

    Variables de entrada (normalizadas)
    ------------------------------------
    idx  nombre        descripción                  unidad
    0    T_geo         Temperatura fluido geotérmico  [°C]
    1    m_geo         Caudal másico geotérmico        [kg/s]
    2    T_amb         Temperatura ambiente            [°C]
    3    f_wf_0        Propiedad fluido trabajo 1     (e.g. T_crit [°C])
    4    f_wf_1        Propiedad fluido trabajo 2     (e.g. P_crit [bar])
    5    P_geo         Presión geotérmica (decisión)  [bar]

    Salidas
    -------
    - W_net     : potencia eléctrica neta [kW]
    - m_wf      : caudal fluido de trabajo [kg/s]
    - T_geo_out : temperatura salida fluido geotérmico [°C]

    Parámetros
    ----------
    n_inputs     : número de entradas (default 6)
    n_hidden     : neuronas por capa oculta (default 64)
    n_layers     : número de bloques residuales (default 4)
    activation   : función de activación ('tanh', 'sin', 'swish', 'adaptive')
    p_geo_bounds : (P_min, P_max) límites de presión geotérmica [bar]
    """

    def __init__(
        self,
        n_inputs: int = 6,
        n_hidden: int = 64,
        n_layers: int = 4,
        activation: str = "tanh",
        p_geo_bounds: tuple = (5.0, 30.0),
    ):
        super().__init__()

        self.p_geo_bounds = p_geo_bounds
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        # --- Capa de entrada ---
        self.input_layer = nn.Linear(n_inputs, n_hidden)

        # --- Bloques residuales ---
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_hidden, activation) for _ in range(n_layers)]
        )

        # --- Normalización de capas ---
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(n_hidden) for _ in range(n_layers)]
        )

        # --- Cabezas de salida ---
        # W_net: potencia neta (salida principal, siempre positiva)
        self.head_wnet = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
            nn.Softplus(),   # garantiza W_net > 0
        )

        # m_wf: caudal fluido de trabajo (siempre positivo)
        self.head_mwf = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
            nn.Softplus(),
        )

        # T_geo_out: temperatura salida fluido geotérmico
        self.head_tgeo_out = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
        )

        # --- Inicialización de pesos ---
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialización Xavier para capas lineales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Propagación hacia adelante.

        Parámetros
        ----------
        x : Tensor [batch, n_inputs]
            Variables de entrada normalizadas.

        Retorna
        -------
        dict con:
            'W_net'      : potencia neta [kW]
            'm_wf'       : caudal fluido de trabajo [kg/s]
            'T_geo_out'  : temperatura salida fluido geotérmico [°C]
            'features'   : representación latente (para análisis)
        """
        # Proyección de entrada
        h = torch.tanh(self.input_layer(x))

        # Bloques residuales con normalización de capas
        for block, norm in zip(self.res_blocks, self.layer_norms):
            h = norm(block(h))

        # Cabezas de salida
        W_net = self.head_wnet(h).squeeze(-1)
        m_wf = self.head_mwf(h).squeeze(-1)
        T_geo_out = self.head_tgeo_out(h).squeeze(-1)

        return {
            "W_net": W_net,
            "m_wf": m_wf,
            "T_geo_out": T_geo_out,
            "features": h,
        }

    def predict_optimal_pressure(
        self,
        T_geo: float,
        m_geo: float,
        T_amb: float,
        f_wf: list,
        normalizer,
        n_points: int = 200,
        device: str = "cpu",
    ) -> dict:
        """
        Barrera de optimización: evalúa la red en una grilla de presiones
        y retorna la presión óptima (máxima W_net).

        Parámetros
        ----------
        T_geo, m_geo, T_amb : condiciones operativas
        f_wf                : propiedades del fluido de trabajo
        normalizer          : objeto DataNormalizer para escalar entradas
        n_points            : puntos en la grilla de presión
        device              : dispositivo de cómputo

        Retorna dict con: P_opt, W_net_opt, grilla completa
        """
        self.eval()
        P_min, P_max = self.p_geo_bounds
        P_grid = torch.linspace(P_min, P_max, n_points, device=device)

        results = []
        with torch.no_grad():
            for P in P_grid:
                inputs = [T_geo, m_geo, T_amb] + list(f_wf) + [P.item()]
                x_raw = torch.tensor(inputs, dtype=torch.float32, device=device).unsqueeze(0)
                x_norm = normalizer.normalize(x_raw)
                out = self.forward(x_norm)
                results.append({
                    "P_geo": P.item(),
                    "W_net": out["W_net"].item(),
                    "m_wf": out["m_wf"].item(),
                    "T_geo_out": out["T_geo_out"].item(),
                })

        # Encontrar máximo de W_net
        best = max(results, key=lambda r: r["W_net"])
        return {"optimal": best, "grid": results}

    def count_parameters(self) -> int:
        """Cuenta los parámetros entrenables de la red."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

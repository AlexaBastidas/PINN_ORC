"""
data/orc_data_generator.py
===========================
Generador de datos sintéticos para el simulador ORC geotérmico.

En la implementación real este módulo se reemplaza por:
1. Datos de un simulador termodinámico (Aspen HYSYS, EES, DWSIM, etc.)
2. Datos de campo de una planta ORC real.

Aquí se genera un dataset sintético físicamente consistente
para validar la arquitectura de la PINN.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Simulador ORC simplificado (reemplazar por el simulador real)
# ---------------------------------------------------------------------------

def simulate_orc(
    T_geo: float,    # Temperatura fluido geotérmico [°C]
    m_geo: float,    # Caudal másico geotérmico [kg/s]
    T_amb: float,    # Temperatura ambiente [°C]
    f_wf: list,      # Propiedades del fluido de trabajo [T_crit, P_crit]
    P_geo: float,    # Presión geotérmica [bar]
    eta_exp: float = 0.80,
    eta_pump: float = 0.75,
    eta_gen: float = 0.95,
    cp_geo: float = 4.18,
    add_noise: bool = True,
    noise_level: float = 0.02,
) -> dict:
    """
    Simula el desempeño del ORC para un conjunto de condiciones.

    Modelo simplificado de fluido orgánico (tipo R245fa):
    - T_sat ≈ 30 * ln(P) + 20  [°C]
    - h_fg ≈ 180 kJ/kg
    - h_liq = cp_wf * T_sat
    - w_exp ≈ 50 * ln(P / P_cond)  [kJ/kg]

    Retorna
    -------
    dict con W_net, m_wf, T_geo_out, Q_evap, Q_cond, COP
    """
    cp_wf_liq = 1.5     # kJ/(kg·K)
    h_fg = 180.0        # kJ/kg
    P_cond = 3.0        # bar (presión de condensación)

    # Temperatura de saturación del fluido de trabajo
    T_sat_wf = 30.0 * np.log(max(P_geo, 1e-3)) + 20.0

    # Verificar que el fluido geotérmico tiene suficiente temperatura
    if T_geo < T_sat_wf + 5.0:
        return None  # condición inválida

    # Entalpías
    h_g = cp_wf_liq * T_sat_wf + h_fg          # vapor saturado en evaporador
    h_f_cond = cp_wf_liq * (T_amb + 8.0)        # líquido saturado en condensador

    # Trabajo del expansor (isentrópico simplificado)
    w_exp_isen = 50.0 * np.log(max(P_geo / P_cond, 1.0))
    h_exp_out = h_g - w_exp_isen

    # Temperatura de salida del fluido geotérmico (balance energético)
    # Q_geo = m_geo * cp_geo * (T_geo - T_geo_out) = m_wf * (h_g - h_f_cond)
    # Se asume m_wf/m_geo ≈ 0.15 a 0.35 dependiendo de condiciones
    ratio_flow = 0.1 + 0.15 * (T_geo - T_sat_wf - 5.0) / (T_geo - 70.0 + 1e-6)
    ratio_flow = np.clip(ratio_flow, 0.05, 0.4)
    m_wf = m_geo * ratio_flow

    Q_evap = m_wf * (h_g - h_f_cond)
    T_geo_out = T_geo - Q_evap / (m_geo * cp_geo + 1e-6)
    T_geo_out = max(T_geo_out, 60.0)  # limitante físico

    # Trabajo de la bomba
    v_liq = 0.001  # m³/kg
    w_pump = v_liq * (P_geo - P_cond) * 100.0 / eta_pump  # kPa·m³/kg = kJ/kg

    # Potencia neta
    W_exp = m_wf * eta_exp * w_exp_isen
    W_pump = m_wf * w_pump
    W_net = eta_gen * W_exp - W_pump

    # Calor disipado en condensador
    Q_cond = m_wf * (h_exp_out - h_f_cond)

    # Eficiencia térmica
    eta_th = max(W_net / (Q_evap + 1e-6), 0.0)

    if add_noise:
        W_net *= (1.0 + np.random.normal(0, noise_level))
        m_wf *= (1.0 + np.random.normal(0, noise_level))
        T_geo_out *= (1.0 + np.random.normal(0, noise_level * 0.5))

    return {
        "W_net": max(W_net, 0.0),
        "m_wf": max(m_wf, 0.0),
        "T_geo_out": T_geo_out,
        "Q_evap": Q_evap,
        "Q_cond": Q_cond,
        "eta_th": eta_th,
        "T_sat_wf": T_sat_wf,
    }


# ---------------------------------------------------------------------------
# Generador del dataset
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int = 2000,
    T_geo_range: tuple = (100.0, 180.0),
    m_geo_range: tuple = (5.0, 50.0),
    T_amb_range: tuple = (15.0, 35.0),
    P_geo_range: tuple = (5.0, 30.0),
    f_wf_options: list = None,
    seed: int = 42,
) -> dict:
    """
    Genera un dataset sintético de simulaciones ORC.

    Parámetros
    ----------
    n_samples    : número de puntos de datos
    *_range      : rangos (min, max) de cada variable
    f_wf_options : lista de fluidos de trabajo disponibles
                   cada elemento es [T_crit_°C, P_crit_bar]
    seed         : semilla para reproducibilidad

    Retorna
    -------
    dict con arrays numpy de entradas y salidas
    """
    np.random.seed(seed)

    if f_wf_options is None:
        # Propiedades críticas de fluidos orgánicos comunes
        # [T_crit (°C), P_crit (bar)]
        f_wf_options = [
            [154.05, 36.40],   # R245fa
            [101.06, 40.59],   # R134a
            [196.85, 29.41],   # Isopentano
            [187.20, 33.78],   # n-Butano
        ]

    inputs = []
    outputs = []
    skipped = 0

    while len(inputs) < n_samples:
        # Muestreo aleatorio de variables
        T_geo = np.random.uniform(*T_geo_range)
        m_geo = np.random.uniform(*m_geo_range)
        T_amb = np.random.uniform(*T_amb_range)
        P_geo = np.random.uniform(*P_geo_range)
        f_wf = f_wf_options[np.random.randint(len(f_wf_options))]

        result = simulate_orc(T_geo, m_geo, T_amb, f_wf, P_geo)

        if result is None or result["W_net"] <= 0:
            skipped += 1
            continue

        inputs.append([T_geo, m_geo, T_amb, f_wf[0], f_wf[1], P_geo])
        outputs.append([result["W_net"], result["m_wf"], result["T_geo_out"]])

    print(f"Dataset generado: {len(inputs)} muestras válidas ({skipped} descartadas)")

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "feature_names": ["T_geo", "m_geo", "T_amb", "f_wf_Tcrit", "f_wf_Pcrit", "P_geo"],
        "output_names": ["W_net", "m_wf", "T_geo_out"],
        "n_samples": len(inputs),
    }


# ---------------------------------------------------------------------------
# Normalizador de datos
# ---------------------------------------------------------------------------

class DataNormalizer:
    """
    Normaliza entradas y salidas al rango [0,1] o z-score.

    Parámetros
    ----------
    method : 'minmax' o 'zscore'
    """

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray = None):
        """Calcula estadísticas de normalización."""
        if self.method == "minmax":
            self.X_min = X.min(axis=0)
            self.X_max = X.max(axis=0)
            self.X_range = self.X_max - self.X_min + 1e-8
            if Y is not None:
                self.Y_min = Y.min(axis=0)
                self.Y_max = Y.max(axis=0)
                self.Y_range = self.Y_max - self.Y_min + 1e-8
        elif self.method == "zscore":
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
            if Y is not None:
                self.Y_mean = Y.mean(axis=0)
                self.Y_std = Y.std(axis=0) + 1e-8
        self.fitted = True
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Normaliza entradas."""
        if self.method == "minmax":
            return (X - self.X_min) / self.X_range
        return (X - self.X_mean) / self.X_std

    def transform_Y(self, Y: np.ndarray) -> np.ndarray:
        """Normaliza salidas."""
        if self.method == "minmax":
            return (Y - self.Y_min) / self.Y_range
        return (Y - self.Y_mean) / self.Y_std

    def inverse_transform_Y(self, Y_norm: np.ndarray) -> np.ndarray:
        """Desnormaliza salidas."""
        if self.method == "minmax":
            return Y_norm * self.Y_range + self.Y_min
        return Y_norm * self.Y_std + self.Y_mean

    def normalize(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza un tensor de entrada (usado en inferencia)."""
        if self.method == "minmax":
            x_min = torch.tensor(self.X_min, dtype=torch.float32)
            x_range = torch.tensor(self.X_range, dtype=torch.float32)
            return (x_tensor - x_min) / x_range
        x_mean = torch.tensor(self.X_mean, dtype=torch.float32)
        x_std = torch.tensor(self.X_std, dtype=torch.float32)
        return (x_tensor - x_mean) / x_std

    def denormalize_wnet(self, W_norm: torch.Tensor) -> torch.Tensor:
        """Desnormaliza la potencia neta (primera salida)."""
        if self.method == "minmax":
            return W_norm * self.Y_range[0] + self.Y_min[0]
        return W_norm * self.Y_std[0] + self.Y_mean[0]


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class ORCDataset(Dataset):
    """
    Dataset PyTorch para entrenamiento de la PINN-ORC.

    Parámetros
    ----------
    inputs_norm   : entradas normalizadas [N, n_inputs]
    outputs_norm  : salidas normalizadas [N, n_outputs]
    inputs_raw    : entradas en escala original [N, n_inputs]
    outputs_raw   : salidas en escala original [N, n_outputs]
    """

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
        inputs_raw: np.ndarray,
        outputs_raw: np.ndarray,
    ):
        self.X_norm = torch.tensor(inputs_norm, dtype=torch.float32)
        self.Y_norm = torch.tensor(outputs_norm, dtype=torch.float32)
        self.X_raw = torch.tensor(inputs_raw, dtype=torch.float32)
        self.Y_raw = torch.tensor(outputs_raw, dtype=torch.float32)

    def __len__(self):
        return len(self.X_norm)

    def __getitem__(self, idx):
        return {
            "x_norm": self.X_norm[idx],
            "y_norm": self.Y_norm[idx],
            "x_raw": self.X_raw[idx],
            "y_raw": self.Y_raw[idx],
        }


def prepare_dataloaders(
    dataset_dict: dict,
    normalizer: DataNormalizer,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple:
    """
    Divide el dataset en train/val/test y crea los DataLoaders.

    Retorna
    -------
    (train_loader, val_loader, test_loader, normalizer)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = dataset_dict["inputs"]
    Y = dataset_dict["outputs"]
    n = len(X)

    # Mezclar índices
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    # Ajustar normalizador con datos de entrenamiento
    normalizer.fit(X[idx_train], Y[idx_train])

    # Crear splits normalizados
    def make_dataset(idx_split):
        X_norm = normalizer.transform_X(X[idx_split])
        Y_norm = normalizer.transform_Y(Y[idx_split])
        return ORCDataset(X_norm, Y_norm, X[idx_split], Y[idx_split])

    train_ds = make_dataset(idx_train)
    val_ds = make_dataset(idx_val)
    test_ds = make_dataset(idx_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Split: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    return train_loader, val_loader, test_loader, normalizer

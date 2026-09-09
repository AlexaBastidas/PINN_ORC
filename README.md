# PINN-ORC: Red Neuronal Informada por la Física para Optimización del Ciclo Rankine Orgánico Geotérmico

## Descripción

Implementación de una PINN que **modela y optimiza simultáneamente** una planta ORC alimentada con fluido geotérmico, incorporando la optimización directamente dentro de la función de pérdida del entrenamiento.

### Formulación matemática

```
L_total = λ1·L_data + λ2·L_physics + λ3·L_opt
```

| Término | Descripción |
|---|---|
| `L_data` | MSE vs. datos del simulador ORC |
| `L_physics` | Residuos de balances de masa/energía y restricciones termodinámicas |
| `L_opt = -W_net` | Maximiza potencia neta durante el entrenamiento |

## Estructura del proyecto

```
pinn_orc/
├── main.py                     # Script principal
├── physics/
│   └── orc_physics.py          # Ecuaciones físicas del ORC (PyTorch)
├── models/
│   ├── pinn_orc.py             # Arquitectura PINN con bloques residuales
│   ├── loss_functions.py       # Función de pérdida compuesta
│   └── trainer.py              # Ciclo de entrenamiento + Early Stopping
├── data/
│   └── orc_data_generator.py   # Simulador ORC sintético + DataNormalizer
└── utils/
    └── evaluation.py           # Métricas y visualizaciones
```

## Variables del modelo

| Índice | Variable | Descripción | Unidad |
|---|---|---|---|
| 0 | `T_geo` | Temperatura fluido geotérmico | °C |
| 1 | `m_geo` | Caudal másico geotérmico | kg/s |
| 2 | `T_amb` | Temperatura ambiente | °C |
| 3 | `f_wf[0]` | T_crítica del fluido de trabajo | °C |
| 4 | `f_wf[1]` | P_crítica del fluido de trabajo | bar |
| 5 | `P_geo` | **Variable de decisión**: presión geotérmica | bar |

**Salidas**: `W_net` [kW], `m_wf` [kg/s], `T_geo_out` [°C]

## Ejecución

```bash
pip install torch numpy matplotlib scipy
python main.py
```

## Restricciones físicas implementadas

- ✅ Balance energético en el evaporador
- ✅ Pinch point mínimo en evaporador (ΔT ≥ 5 K)
- ✅ Temperatura mínima de salida del fluido geotérmico (≥ 70 °C)
- ✅ Título mínimo de vapor a la salida del expansor (≥ 0.90)
- ✅ Potencia neta positiva
- ✅ Límites operativos de presión (P_min ≤ P_geo ≤ P_max)

## Próximos pasos

1. **Reemplazar** `simulate_orc()` con datos reales o un simulador como EES/DWSIM
2. **Integrar CoolProp** para propiedades termodinámicas precisas del fluido de trabajo
3. **Ajustar pesos** λ1, λ2, λ3 mediante búsqueda de hiperparámetros
4. **Validar** con datos de campo de la planta ORC real

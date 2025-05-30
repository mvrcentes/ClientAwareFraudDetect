# 🕵️‍♂️ Fraud Detection: High-Amount Anomalies per Client

Este proyecto tiene como objetivo detectar transacciones fraudulentas de **alto monto**, inusuales según el historial de cada cliente, utilizando técnicas de aprendizaje automático e ingeniería de variables comportamentales y temporales.

📄 [**Ver informe técnico completo (PDF)**](./Reporte%20final.pdf)

---

## 📊 Resumen

Se diseñó un pipeline de detección de fraudes que incluye:

- Análisis exploratorio (EDA) para entender la distribución de montos y el desbalance de clases.
- Ingeniería de variables con señales como horarios nocturnos, Z-score por cliente y comportamiento de compra.
- Entrenamiento de un modelo LightGBM con partición temporal y undersampling del 30 % en no-fraudes.
- Evaluación con métricas estándar (AUC-ROC, F1) y personalizadas.
- Optimización de hiperparámetros con **Optuna**.
- Comparación de estrategias de evaluación para seleccionar la métrica más adecuada.

---

## 🗃️ Dataset

- **Tipo:** Simulado
- **Rango:** Enero 2019 – Diciembre 2020
- **Tamaño:** ~1.85 millones de transacciones
- **Variable objetivo:** `is_fraud` (0: legítima, 1: fraudulenta)
- **Testing:** Último trimestre de 2020 (división temporal)

---

## ⚙️ Metodología

### 1. EDA

- Menos del 1 % de las transacciones son fraudes.
- Se observaron montos altos más asociados a fraude.
- El **80 % de los fraudes** ocurre entre 00:00 y 06:00.

### 2. Ingeniería de Variables

- Variables temporales (`is_night`, `is_weekend`, `hour`, etc.).
- Z-score del monto por cliente para capturar anomalías.
- Indicadores de comportamiento: primera visita, frecuencia de compras, distancia cliente–comercio.

### 3. Modelo Base

- **Modelo:** LightGBM
- **Partición temporal:** Train: antes de octubre 2020 / Test: oct–dic 2020
- **Balanceo:** Undersampling al 30 % de no-fraudes

### 4. Métricas Personalizadas

Se evaluaron tres estrategias de evaluación:

| Métrica           | Propósito                           | Valor      |
| ----------------- | ----------------------------------- | ---------- |
| `fp_penalty`      | Penalizar falsos positivos          | 1.1235     |
| `amt_weighted_fp` | Penalizar errores según monto       | 1.1545     |
| `balanced_f1` ✅  | Equilibrio entre precisión y recall | **0.8356** |

> Se seleccionó `balanced_f1` como métrica final por su solidez práctica y equilibrio.

### 5. Optimización

- **Herramienta:** [Optuna](https://optuna.org/)
- **Trials:** 50
- **Objetivo:** Maximizar AUC-ROC
- Se probaron combinaciones de hiperparámetros como `learning_rate`, `scale_pos_weight`, `num_leaves`, entre otros.

---

## 🧪 Resultados

- **AUC final:** 0.99955
- **F1-Score:** 0.8356
- **balanced_f1:** 0.8356
- **Hallazgo clave:** is_night fue una variable altamente predictiva.

---

## 💻 Implementación

- **Lenguaje:** Python 3.11
- **Entorno:** Jupyter Notebook
- **Librerías principales:** `pandas`, `numpy`, `lightgbm`, `optuna`, `scikit-learn`, `seaborn`, `matplotlib`

📂 Archivos relevantes:

- `FraudDetection_HighAmountAnomalies.ipynb`: pipeline completo
- `main.py`: ejecución por lotes
- `feature_engineering_work/`: transformaciones auxiliares
- `Reporte final.pdf`: informe entregable

---

## 🧠 Conclusiones

> El enfoque centrado en el comportamiento individual del cliente y el uso de métricas personalizadas permitió construir un modelo eficaz, adaptable y útil para escenarios reales donde se requiere un equilibrio entre detección de fraudes y control de alertas falsas.

---

## 🔗 Repositorio

📁 Repositorio completo del proyecto: [https://github.com/mvrcentes/ClientAwareFraudDetect](https://github.com/mvrcentes/ClientAwareFraudDetect)

---

## 📥 Contacto

**Marco Ramírez**  
Estudiante de TI enfocado en análisis de datos, seguridad y sistemas inteligentes  
📧 mvrcentes@gmail.com

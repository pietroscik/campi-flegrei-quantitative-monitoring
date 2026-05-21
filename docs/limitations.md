# Limitations — Campi Flegrei Monitoring System

## 1. Predictive Limitations

The system does not provide deterministic eruption prediction. All outputs represent probabilistic or statistical indicators of seismic activity.

---

## 2. Model Simplifications

The ETAS model assumes:
- purely seismic self-excitation
- no coupling with geochemical or deformation data
- simplified parameter constancy within calibration windows

---

## 3. Data Bias

Seismic catalogs may suffer from:
- magnitude of completeness (Mc) variability
- temporal changes in detection sensitivity
- reporting delays or inconsistencies

---

## 4. Spatial Resolution

The analysis is performed on a 2D geographic projection and does not incorporate full 3D crustal structure.

---

## 5. Parameter Sensitivity

ETAS and b-value estimates are sensitive to:
- window size selection
- magnitude thresholds
- smoothing parameters

---

## 6. Lack of External Covariates

The current framework does not include:
- ground deformation (GPS / InSAR)
- gas emissions
- thermal anomalies

This limits physical interpretability.

---

## 7. Real-Time Constraints

Although structured as a live system, updates are batch-based and not true streaming in the physical sense.

---

## 8. Operational Use

The system is intended for research and educational purposes and not for operational civil protection decision-making.

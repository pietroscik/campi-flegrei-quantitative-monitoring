# Assumptions — Campi Flegrei Monitoring System

## 1. Data Completeness

The seismic catalog is assumed to be complete above a minimum magnitude threshold (Mc). Below this threshold, detection bias may affect statistical stability.

---

## 2. Spatial Homogeneity

The model assumes spatial homogeneity within the selected bounding box of the Campi Flegrei caldera. Local heterogeneities are not explicitly modeled.

---

## 3. Stationarity within Windows

Rolling estimations (b-value, ETAS parameters) assume local stationarity within each time window.

---

## 4. Independence of Signal Components

The multi-signal unrest index assumes partial independence between:
- b-value variations
- ETAS intensity
- seismic rate changes

This is a simplifying assumption for aggregation.

---

## 5. ETAS Model Structure

The ETAS formulation assumes:
- triggering is driven by past seismicity only
- no external geophysical covariates are included
- parameter stability within calibration intervals

---

## 6. Detection Bias

No explicit correction is applied for:
- network sensitivity changes over time
- catalog incompleteness variations

---

## 7. Interpretability

The system is designed as a statistical monitoring tool and not as a deterministic eruption predictor.

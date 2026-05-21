# 🌋 Campi Flegrei Quantitative Monitoring System

## Abstract

This repository implements a quantitative seismic monitoring framework for the Campi Flegrei volcanic caldera (Southern Italy). The system integrates statistical seismology, stochastic point-process modeling, and multi-indicator anomaly detection to simulate a near-real-time volcanic unrest monitoring environment.

The framework combines the Gutenberg–Richter law (b-value analysis), the ETAS (Epidemic-Type Aftershock Sequence) model, and a composite unrest index derived from multi-signal fusion. The objective is not deterministic eruption forecasting, but probabilistic characterization of seismic dynamics and system-level unrest detection.

---

## 1. Introduction

Volcanic calderas such as Campi Flegrei exhibit complex seismic behavior driven by coupled mechanical, thermal, and fluid-dynamic processes. Traditional single-metric approaches are insufficient to capture the full dynamics of unrest evolution.

This system proposes an integrated computational pipeline that:
- quantifies seismicity evolution
- models triggering dynamics
- detects statistical anomalies
- produces interpretable alert states

---

## 2. Data

The analysis is based on seismic event catalogs provided by INGV (Istituto Nazionale di Geofisica e Vulcanologia).

Each event includes:
- origin time
- magnitude
- geographic coordinates
- depth (when available)

A spatial bounding box is applied to isolate the Campi Flegrei caldera region.

---

## 3. Methodology

### 3.1 Gutenberg–Richter Law

Seismicity magnitude distribution is modeled using:

- log N(M) = a − bM

The b-value is estimated using rolling windows to capture temporal variations in stress regimes.

---

### 3.2 ETAS Model

The ETAS model represents seismicity as a self-exciting point process:

- background rate (μ)
- triggered seismicity (K)
- magnitude scaling (α)
- temporal decay parameters (p, c)

The conditional intensity λ(t) describes the instantaneous seismic hazard.

---

### 3.3 Multi-Signal Unrest Index

A composite indicator is constructed by integrating:

- b-value anomalies
- ETAS intensity variations
- seismic rate fluctuations

All signals are normalized and aggregated into a single unrest metric.

---

### 3.4 Early Warning System

The unrest index is mapped into discrete operational states:

- NORMAL
- ELEVATED
- CRITICAL

Thresholds are empirically defined and can be recalibrated.

---

## 4. System Architecture

The system is structured in modular layers:

- **Core Layer**: seismic modeling and statistical computation
- **Pipeline Layer**: orchestration of analysis workflow
- **Service Layer**:
  - FastAPI (data access layer)
  - Streamlit (visual analytics dashboard)
  - Worker (batch execution engine)

---

## 5. Outputs

The system generates:

- cleaned seismic catalog
- rolling b-value time series
- ETAS conditional intensity λ(t)
- multi-signal unrest index
- early warning classification timeline

---

## 6. Key Assumptions

- completeness of seismic catalog above magnitude threshold
- local stationarity within rolling windows
- partial independence between signal components
- absence of external geophysical covariates (e.g., deformation, gas emissions)

---

## 7. Limitations

- not a deterministic eruption forecasting system
- sensitivity to parameter calibration (ETAS, window size)
- no integration of geodetic or geochemical data
- simplified spatial representation (2D projection)
- batch-based updates rather than true real-time streaming

---

## 8. Intended Use

This system is intended for:
- research in statistical volcanology
- educational purposes in geophysical data science
- prototyping of monitoring systems

It is not intended for operational civil protection decision-making.

---

## 9. Conclusion

This repository provides a unified computational framework for volcanic seismicity analysis, combining classical seismological theory with modern probabilistic modeling and system-level integration. It demonstrates how multi-signal statistical inference can be operationalized into a coherent monitoring architecture.

---

## 10. References (conceptual)

- Gutenberg & Richter (frequency–magnitude relation)
- Ogata (1988) ETAS model formulation
- Classical statistical seismology literature

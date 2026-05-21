# Methodology — Campi Flegrei Quantitative Monitoring System

## 1. Overview

This project implements a statistical–computational framework for the analysis of volcanic seismicity in the Campi Flegrei caldera. The system integrates classical seismological laws with stochastic point-process modeling and multi-signal risk indicators.

The pipeline is designed to be reproducible, modular, and suitable for near-real-time monitoring simulations.

---

## 2. Data Source

Seismic events are retrieved from the INGV (Istituto Nazionale di Geofisica e Vulcanologia) catalog.

Each event includes:
- origin time
- magnitude (ML / Mw where available)
- latitude and longitude
- depth (when available)

---

## 3. Gutenberg–Richter Law (b-value)

The frequency–magnitude distribution is modeled as:

- log N(M) = a − bM

The b-value is estimated using rolling windows to capture temporal variations in stress regime and crustal heterogeneity.

---

## 4. ETAS Model

The Epidemic-Type Aftershock Sequence (ETAS) model is used to estimate conditional seismic intensity:

- background seismicity (μ)
- aftershock productivity (K)
- magnitude scaling (α)
- temporal decay (p, c)

The conditional intensity function λ(t) represents the instantaneous seismic hazard.

---

## 5. Multi-Signal Unrest Index

A composite index is constructed using:

- normalized b-value anomalies
- ETAS intensity deviations
- seismic rate changes

The signals are aggregated into a single unrest indicator through weighted normalization.

---

## 6. Early Warning System

A rule-based classification system maps the unrest index into discrete states:

- NORMAL
- ELEVATED
- CRITICAL

Thresholds are empirically defined and can be recalibrated.

---

## 7. Output Layer

The system produces:
- processed seismic catalog
- rolling b-value series
- ETAS intensity time series
- unrest index
- alert state timeline

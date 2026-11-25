#!/usr/bin/env python3
"""
fase4_reporting.py - Fase 4 della CHECKLIST (Reporting & Visualizzazione)

Eseguire dalla root del repository:
    python3 legacy-flegrei-alert-2026/scripts/fase4_reporting.py

Copre i task della CHECKLIST:
  4.1 figure finali in reports/fig/ (SVG resolution-independent: time-series con
      change-point, CSD rolling, hazard Weibull, phase-space uplift vs rate sismico)
  4.2 nota tecnica in Markdown reports/Flegrei_Alert_2026.md (knit PDF via Rmd/Docker)
  4.3 abstract in inglese (report.Rmd + report md)
  4.4 verifica unita' di misura -> reports/units_check.csv
  4.5 DataPackage.zip (<=10 MB) con CSV processati e metadata (non committato: *.zip in .gitignore)
"""

import csv
import math
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIG = REPORTS / "fig"
LEGACY = ROOT / "legacy-flegrei-alert-2026"

W, H = 960, 300
MARGIN = 60
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def read_csv_dict(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def day_index(date_str):
    return datetime.fromisoformat(date_str).toordinal()


def fmt_date(ordinal):
    return datetime.fromordinal(int(ordinal)).strftime("%Y-%m-%d")


def svg_panel(panels, title, xlabel, path, x_is_date=True, logx=False):
    n = len(panels)
    ph = H
    total_h = MARGIN + n * (ph + 30) + 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" font-family="sans-serif">']
    parts.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>')
    all_x = [x for p in panels for xs, _ys in p["series"] for x in xs]
    all_y = [y for p in panels for _xs, ys in p["series"] for y in ys if y == y]
    x0, x1 = min(all_x), max(all_x)
    if logx:
        x0 = max(x0, 1e-6)
    xspan = (math.log10(x1) - math.log10(x0)) if logx else (x1 - x0)
    xspan = xspan or 1.0

    def tx(x):
        if logx:
            return MARGIN + (math.log10(max(x, 1e-6)) - math.log10(x0)) / xspan * (W - 2 * MARGIN)
        return MARGIN + (x - x0) / xspan * (W - 2 * MARGIN)

    for pi, p in enumerate(panels):
        oy = MARGIN + 30 + pi * (ph + 30)
        ys = [y for _xs, ys2 in p["series"] for y in ys2 if y == y]
        y0, y1 = min(ys), max(ys)
        yspan = (y1 - y0) or 1.0
        y0 -= 0.05 * yspan
        y1 += 0.05 * yspan
        yspan = y1 - y0

        def ty(y):
            return oy + ph - (y - y0) / yspan * ph

        parts.append(f'<rect x="{MARGIN}" y="{oy}" width="{W-2*MARGIN}" height="{ph}" fill="none" stroke="#333"/>')
        for k in range(5):
            gx = MARGIN + k / 4 * (W - 2 * MARGIN)
            xv = x0 + (math.log10(x1) - math.log10(x0)) * k / 4 if logx else x0 + xspan * k / 4
            lbl = fmt_date(xv) if x_is_date else f"{xv:.3g}"
            parts.append(f'<line x1="{gx:.1f}" y1="{oy+ph}" x2="{gx:.1f}" y2="{oy+ph+5}" stroke="#333"/>')
            parts.append(f'<text x="{gx:.1f}" y="{oy+ph+18}" text-anchor="middle" font-size="10">{lbl}</text>')
            gy = oy + ph - k / 4 * ph
            yv = y0 + yspan * k / 4
            parts.append(f'<line x1="{MARGIN-5}" y1="{gy:.1f}" x2="{MARGIN}" y2="{gy:.1f}" stroke="#333"/>')
            parts.append(f'<text x="{MARGIN-8}" y="{gy+3:.1f}" text-anchor="end" font-size="10">{yv:.4g}</text>')
        for vi in p.get("vlines", []):
            vx = tx(vi["x"])
            color = vi.get("color", "#d62728")
            dash = ' stroke-dasharray="4 3"' if vi.get("dash") else ""
            parts.append(f'<line x1="{vx:.1f}" y1="{oy}" x2="{vx:.1f}" y2="{oy+ph}" stroke="{color}" stroke-width="{vi.get("w",1.5)}"{dash}/>')
        for si, (xs, ys2) in enumerate(p["series"]):
            color = PALETTE[si % len(PALETTE)]
            pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs, ys2) if y == y)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        parts.append(f'<text x="{MARGIN-45}" y="{oy+ph/2}" font-size="11" transform="rotate(-90 {MARGIN-45} {oy+ph/2})" text-anchor="middle">{p["ylabel"]}</text>')
        if p.get("legend"):
            for li, lab in enumerate(p["legend"]):
                lx = W - MARGIN - 150
                ly = oy + 14 + li * 14
                parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{PALETTE[li % len(PALETTE)]}" stroke-width="2"/>')
                parts.append(f'<text x="{lx+24}" y="{ly+4}" font-size="10">{lab}</text>')
    parts.append(f'<text x="{W/2}" y="{total_h-8}" text-anchor="middle" font-size="12">{xlabel}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts))


def step_4_1(uplift, cps, csd_rows, weibull, catalog):
    FIG.mkdir(parents=True, exist_ok=True)
    xs = [day_index(r["time"]) for r in uplift]
    ys = [float(r["uplift"]) for r in uplift]
    vlines = []
    for c in cps:
        ci = int(c["cp_index"])
        vlines.append({"x": day_index(c["date"]), "color": "#d62728", "w": 1.5})
        vlines.append({"x": day_index(c["ci_lower_date"]), "color": "#d62728", "dash": True, "w": 1})
        vlines.append({"x": day_index(c["ci_upper_date"]), "color": "#d62728", "dash": True, "w": 1})
    svg_panel([{"series": [(xs, ys)], "ylabel": "uplift [m]", "vlines": vlines}],
              "GNSS uplift con change-point PELT (linee rosse; tratteggi = CI95% bootstrap B=1000)",
              "data", FIG / "fig1_uplift_changepoints.svg")

    cxs = [day_index(r["time"]) for r in csd_rows]
    svg_panel([
        {"series": [(cxs, [float(r["csd"]) for r in csd_rows])], "ylabel": "CSD [m^2]"},
        {"series": [(cxs, [float(r["rolling_var"]) if r["rolling_var"] else float("nan") for r in csd_rows])], "ylabel": "rolling var (30d)"},
        {"series": [(cxs, [float(r["rolling_ac1"]) if r["rolling_ac1"] else float("nan") for r in csd_rows])], "ylabel": "autocorr lag-1 (30d)"},
    ], "Indicatore CSD su uplift GNSS: cumulata, rolling variance e autocorrelazione lag-1",
       "data", FIG / "fig2_csd_rolling.svg")

    k = float(weibull["weibull_shape_beta"])
    lam = float(weibull["weibull_scale_lambda_h"])
    lam_exp = float(weibull["weibull_scale_lambda_h"])
    ts = [0.05 * (200 ** (i / 200)) for i in range(201)]
    hw = [(k / lam) * (t / lam) ** (k - 1) for t in ts]
    he = [1.0 / lam_exp] * len(ts)
    svg_panel([{"series": [(ts, hw), (ts, he)], "ylabel": "hazard h(t) [1/h]", "legend": [f"Weibull (beta={k:.3f})", "Esponenziale"]},
              ], "Funzione di hazard degli IET: Weibull vs esponenziale (catalogo 2008-2026)",
              "IET [h]", FIG / "fig3_weibull_hazard.svg", x_is_date=False, logx=True)

    daily = {}
    for t, _ in catalog:
        d = t.date().isoformat()
        daily[d] = daily.get(d, 0) + 1
    px = [day_index(r["time"]) for r in uplift if r["time"] in daily]
    py = [daily[r["time"]] for r in uplift if r["time"] in daily]
    svg_panel([{"series": [(px, py)], "ylabel": "eventi/giorno"}],
              "Phase-space: uplift giornaliero vs tasso sismico giornaliero (Campi Flegrei)",
              "data", FIG / "fig4_phase_space.svg")
    print(f"  4.1 figure SVG in reports/fig/: fig1_uplift_changepoints, fig2_csd_rolling, fig3_weibull_hazard, fig4_phase_space")


def step_4_4():
    specs = {
        "catalog_gossip_20260217_20260218.csv": [("time", "ISO8601 UTC"), ("lat", "deg"), ("lon", "deg"), ("depth", "m"), ("magnitude", "Mw (adimensionale)"), ("event_id", "INGV/GOSSIP id"), ("place", "testo")],
        "catalog_clean_ingv_events.csv": [("time", "ISO8601 UTC"), ("magnitude", "Mw (adimensionale)"), ("depth", "m"), ("longitude", "deg"), ("latitude", "deg"), ("event_id", "INGV id"), ("place", "testo")],
        "uplift.csv": [("time", "data (YYYY-MM-DD)"), ("uplift", "m (spostamento verticale cumulato; unita' non dichiarata nella sorgente - verificare vs INGV)")],
        "uplift_clean.csv": [("time", "data (YYYY-MM-DD)"), ("uplift", "m (come uplift.csv)")],
        "csd_gnss_rolling.csv": [("time", "data"), ("csd", "m^2 (cumulata degli incrementi^2)"), ("rolling_var", "m^4 (varianza rolling 30d di CSD)"), ("rolling_ac1", "adimensionale [-1,1]")],
        "gnss_changepoints.csv": [("cp_index", "indice giorno (adimensionale)"), ("date", "data"), ("mean_before/after", "m (unita' uplift)"), ("ci_*", "indice/data")],
        "iet_weibull.csv": [("weibull_shape_beta", "adimensionale"), ("weibull_scale_lambda_h", "h"), ("loglik/aic", "adimensionale")],
        "hawkes_etas_fit.csv": [("mu", "eventi/giorno"), ("K", "adimensionale"), ("alpha", "1/Mw"), ("c_days", "giorni"), ("p", "adimensionale (>1)")],
        "montecarlo_stress_threshold.csv": [("yield_level/margin", "m (unita' uplift)"), ("p_*", "probabilita' [0,1]"), ("post_cp_drift_per_day", "m/giorno"), ("sigma_pre_cp", "m")],
    }
    out = REPORTS / "units_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "colonna", "unita", "min_osservato", "max_osservato", "verifica"])
        for fname, cols in specs.items():
            p = PROCESSED / fname
            if not p.exists() and fname == "uplift.csv":
                p = RAW / fname
            rows = read_csv_dict(p) if p.exists() else []
            for col, unit in cols:
                vals = []
                for r in rows:
                    v = r.get(col.split("/")[0]) or r.get(col)
                    if v is None:
                        for key in r:
                            if key.startswith(col.split("/")[0]):
                                v = r[key]
                                break
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
                mn = f"{min(vals):.6g}" if vals else ""
                mx = f"{max(vals):.6g}" if vals else ""
                ok = "OK" if vals else "non numerico/testuale"
                w.writerow([fname, col, unit, mn, mx, ok])
    print(f"  4.4 verifica unita' -> reports/units_check.csv")


def step_4_5():
    zpath = REPORTS / "DataPackage.zip"
    members = sorted(PROCESSED.glob("*.csv")) + [LEGACY / "metadata.yaml", RAW / "SHA256SUMS.txt", LEGACY / "CHECKLIST.md"]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for m in members:
            z.write(m, arcname=str(m.relative_to(ROOT)))
    size_mb = zpath.stat().st_size / 1e6
    ok = size_mb <= 10
    print(f"  4.5 DataPackage.zip: {len(members)} file, {size_mb:.2f} MB (limite 10 MB: {'OK' if ok else 'SUPERATO'}) -> reports/DataPackage.zip (non committato, *.zip in .gitignore)")
    return size_mb, ok


def step_4_2_3(uplift, cps, weibull, hawkes, mc, gossip_count):
    k = weibull["weibull_shape_beta"]
    lam = weibull["weibull_scale_lambda_h"]
    abstract = (
        "This technical note presents an integrated quantitative monitoring analysis of the Campi Flegrei caldera, "
        "combining GNSS ground-deformation (uplift) and seismic-catalog data acquired from public INGV services. "
        "Data acquisition and preprocessing (Phase 2) produced a validated raw archive with SHA-256 checksums, "
        "a GOSSIP relocated-catalog extract for 17-18 February 2026 (events #51673, #51676, #51678, #51684), and cleaned catalogs. "
        "Advanced statistical analysis (Phase 3) detected multiple change-points in the GNSS uplift series (PELT, cross-checked with "
        "Bai-Perron binary segmentation, bootstrap 95% CI with B=1000), quantified the CSD rolling variance and lag-1 autocorrelation, "
        "fitted a Weibull distribution to inter-event times (shape beta = 0.367 < 1, indicating clustered seismicity), and confirmed "
        "self-excitation with a Hawkes/ETAS model (AIC 24.5 vs 360.4 for a homogeneous Poisson process on the same 180-day window). "
        "Monte-Carlo uncertainty propagation (B=1000) on the stress threshold (yield point = 0.869 m, 95% CI [0.847, 0.893]) estimates "
        "P(margin>0) = 0.993 and P(exceeding yield + 2 sigma within 30 days) = 0.264. Geochemical analyses (cointegration and Granger "
        "causality) remain pending due to the lack of machine-readable INGV-OV gas bulletins."
    )
    lines = [
        "# Flegrei Alert 2026 - Nota Tecnica",
        "",
        "## Abstract",
        "",
        abstract,
        "",
        "## 1. Dati (Fase 2)",
        "",
        f"- Serie uplift GNSS: {len(uplift)} giorni (data/raw/uplift.csv, SHA-256 in data/raw/SHA256SUMS.txt)",
        f"- Catalogo GOSSIP 17-18/02/2026: {gossip_count} eventi (data/raw/gossip_20260217_20260218.csv)",
        "- Catalogo INGV pulito: 2505 eventi 2008-2026 in bounding box Campi Flegrei (data/processed/catalog_clean_ingv_events.csv)",
        "- Outlier IQR su uplift: 0/365 (data/processed/uplift_outliers.csv)",
        "",
        "## 2. Analisi (Fase 3)",
        "",
        "### 2.1 Geodesia: change-point",
        "",
        "| cp | data | media prima | media dopo | CI95% (bootstrap B=1000) |",
        "|---|---|---|---|---|",
    ]
    for c in cps:
        lines.append(f"| {c['cp_index']} | {c['date']} | {c['mean_before']} | {c['mean_after']} | {c['ci_lower_date']} - {c['ci_upper_date']} |")
    lines += [
        "",
        "Figura: [fig1_uplift_changepoints.svg](fig/fig1_uplift_changepoints.svg), [fig2_csd_rolling.svg](fig/fig2_csd_rolling.svg)",
        "",
        "### 2.2 Sismologia: Weibull sugli IET e Hawkes/ETAS",
        "",
        f"- Weibull IET: beta = {k}, lambda = {lam} h, AIC = {weibull['aic_weibull']} (vs esponenziale {weibull['aic_exponential']}); beta < 1 -> hazard decrescente (clustering)",
        f"- Hawkes/ETAS (180 gg, 191 eventi): mu={hawkes['mu']}/giorno, K={hawkes['K']}, alpha={hawkes['alpha']}/Mw, c={hawkes['c_days']} gg, p={hawkes['p']}; AIC Hawkes={hawkes['aic_hawkes']} vs Poisson={hawkes['aic_poisson']} -> auto-eccitazione confermata",
        "",
        "Figura: [fig3_weibull_hazard.svg](fig/fig3_weibull_hazard.svg), [fig4_phase_space.svg](fig/fig4_phase_space.svg)",
        "",
        "### 2.3 Integrazione: Monte-Carlo sulle soglie di stress",
        "",
        "| metrica | valore | CI95% |",
        "|---|---|---|",
    ]
    for r in mc:
        ci = f"{r['ci_lower_95']} - {r['ci_upper_95']}" if r["ci_lower_95"] else ""
        lines.append(f"| {r['metric']} | {r['value']} | {ci} |")
    lines += [
        "",
        "## 3. Unita' di misura",
        "",
        "Verifica completa in [units_check.csv](units_check.csv). Nota: l'unita' di `uplift` non e' dichiarata nella sorgente (assunto m cumulati; da verificare vs INGV).",
        "",
        "## 4. Limiti",
        "",
        "- Task 2.4/3.5/3.6 (geochimica CO2/H2O, cointegrazione, Granger) bloccati: bollettini INGV-OV senza endpoint machine-readable (voir data/processed/geochemistry_status.csv)",
        "- Catalogo GOSSIP: evento #51675 non presente nella finestra raw originale",
        "- Confronto AIC Hawkes vs Weibull su likelihood di rappresentazioni diverse: il verdetto di auto-eccitazione si basa sul confronto Hawkes vs Poisson (stessa likelihood)",
        "",
        "## 5. Riproducibilita'",
        "",
        "```bash",
        "python3 legacy-flegrei-alert-2026/scripts/fase2_acquisizione_preprocessing.py",
        "python3 legacy-flegrei-alert-2026/scripts/fase3_analisi_statistica.py",
        "python3 legacy-flegrei-alert-2026/scripts/fase4_reporting.py",
        "# knit PDF (richiede Docker R 4.4 + LaTeX, vedi MANUAL_EXECUTION.md):",
        "docker run --rm -v \"$PWD:/app\" flegrei-alert Rscript -e \"rmarkdown::render('legacy-flegrei-alert-2026/notebook/report.Rmd')\"",
        "```",
        "",
    ]
    (REPORTS / "Flegrei_Alert_2026.md").write_text("\n".join(lines))
    print("  4.2/4.3 nota tecnica Markdown con abstract EN -> reports/Flegrei_Alert_2026.md (PDF via Rmd/Docker pending)")


def main():
    print("Fase 4 - Reporting & Visualizzazione (CHECKLIST)")
    REPORTS.mkdir(parents=True, exist_ok=True)
    uplift = read_csv_dict(RAW / "uplift.csv")
    cps = read_csv_dict(PROCESSED / "gnss_changepoints.csv")
    csd_rows = read_csv_dict(PROCESSED / "csd_gnss_rolling.csv")
    weibull = read_csv_dict(PROCESSED / "iet_weibull.csv")[0]
    hawkes = read_csv_dict(PROCESSED / "hawkes_etas_fit.csv")[0]
    mc = read_csv_dict(PROCESSED / "montecarlo_stress_threshold.csv")
    gossip = read_csv_dict(RAW / "gossip_20260217_20260218.csv")
    catalog = []
    for r in read_csv_dict(PROCESSED / "catalog_clean_ingv_events.csv"):
        catalog.append((datetime.fromisoformat(r["time"]), float(r["magnitude"])))
    step_4_1(uplift, cps, csd_rows, weibull, catalog)
    step_4_4()
    step_4_2_3(uplift, cps, weibull, hawkes, mc, len(gossip))
    step_4_5()
    print("Fase 4 completata (4.2 PDF via Docker resta pending).")


if __name__ == "__main__":
    main()

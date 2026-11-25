#!/usr/bin/env python3
"""
fase5_revisione_invio.py - Fase 5 della CHECKLIST (Revisione & Invio)

Eseguire dalla root del repository:
    python3 legacy-flegrei-alert-2026/scripts/fase5_revisione_invio.py

Copre i task della CHECKLIST:
  5.1 dry-run della catena riproducibile (fase2 -> fase3 -> fase4) + validazione
      artefatti (SHA-256, DataPackage.zip <= 10 MB) -> reports/dry_run_report.md
      (la parte Docker start -> PDF resta pending: richiede daemon Docker + immagine R)
  5.2 bozza di richiesta peer review -> reports/peer_review_request.md (NON inviata)
  5.3 bozza email per Dott. De Martino, Dott. Di Vito, Dott.ssa Pappalardo ->
      reports/email_draft.md (NON inviata: invio manuale previa approvazione)
  5.4 bozza metadati Zenodo -> reports/zenodo_deposit.json (NON caricato:
      publishing su servizio esterno richiede approvazione esplicita)
  5.5/5.6 invio nota tecnica e follow-up: restano manuali (voir bozze)
"""

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
LEGACY = ROOT / "legacy-flegrei-alert-2026"
SCRIPTS = LEGACY / "scripts"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_5_1():
    results = []
    for name in ["fase2_acquisizione_preprocessing.py", "fase3_analisi_statistica.py", "fase4_reporting.py"]:
        r = subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, capture_output=True, text=True)
        results.append((name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""))

    checks = []
    sums = {}
    for line in (RAW / "SHA256SUMS.txt").read_text().strip().splitlines():
        h, fn = line.split(None, 1)
        sums[fn.strip()] = h
    for fn, h in sums.items():
        p = RAW / fn
        ok = p.exists() and sha256_file(p) == h
        checks.append((f"sha256 {fn}", "PASS" if ok else "FAIL"))

    expected = [
        PROCESSED / "catalog_gossip_20260217_20260218.csv",
        PROCESSED / "catalog_clean_ingv_events.csv",
        PROCESSED / "uplift_clean.csv",
        PROCESSED / "gnss_changepoints.csv",
        PROCESSED / "csd_gnss_rolling.csv",
        PROCESSED / "iet_weibull.csv",
        PROCESSED / "hawkes_etas_fit.csv",
        PROCESSED / "montecarlo_stress_threshold.csv",
        REPORTS / "Flegrei_Alert_2026.md",
        REPORTS / "units_check.csv",
        REPORTS / "fig" / "fig1_uplift_changepoints.svg",
        REPORTS / "fig" / "fig2_csd_rolling.svg",
        REPORTS / "fig" / "fig3_weibull_hazard.svg",
        REPORTS / "fig" / "fig4_phase_space.svg",
    ]
    for p in expected:
        ok = p.exists() and p.stat().st_size > 0
        checks.append((f"artefatto {p.relative_to(ROOT)}", "PASS" if ok else "FAIL"))

    zpath = REPORTS / "DataPackage.zip"
    if zpath.exists():
        size_mb = zpath.stat().st_size / 1e6
        with zipfile.ZipFile(zpath) as z:
            bad = z.testzip()
        checks.append((f"DataPackage.zip integrita' ({size_mb:.2f} MB, limite 10 MB)", "PASS" if bad is None and size_mb <= 10 else "FAIL"))
    else:
        checks.append(("DataPackage.zip presente", "FAIL"))

    n_fail = sum(1 for _, s in checks if s == "FAIL")
    lines = [
        "# Dry-run Fase 5 - catena riproducibile",
        "",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Esecuzione script",
        "",
        "| script | esito | ultimo log |",
        "|---|---|---|",
    ]
    for name, rc, last in results:
        lines.append(f"| {name} | {'PASS' if rc == 0 else 'FAIL'} | {last} |")
    lines += ["", "## Validazione artefatti", "", "| check | esito |", "|---|---|"]
    for name, status in checks:
        lines.append(f"| {name} | {status} |")
    lines += [
        "",
        f"**Esito complessivo: {'PASS' if n_fail == 0 else f'{n_fail} FAIL'}**",
        "",
        "Nota: la parte 'Docker start -> PDF' del task 5.1 richiede daemon Docker + immagine R 4.4 (non disponibile in CI sandbox); resta pending e va eseguita via MANUAL_EXECUTION.md §§1-4.",
        "",
    ]
    (REPORTS / "dry_run_report.md").write_text("\n".join(lines))
    print(f"  5.1 dry-run: {len(results)} script, {len(checks)} check, {n_fail} FAIL -> reports/dry_run_report.md")
    return n_fail == 0


def step_5_2():
    text = """# Richiesta di peer review (BOZZA - non inviata)

A: collega geodeta / sismologo (da designare)
Oggetto: Peer review bozza "Flegrei Alert 2026 - Nota Tecnica" (Campi Flegrei)

Ciao,

ti invio in bozza la nota tecnica Flegrei Alert 2026 (reports/Flegrei_Alert_2026.md, figure in reports/fig/, DataPackage.zip) per una revisione rapida (dry-run in reports/dry_run_report.md).

Punti su cui chiedo feedback:
1. Change-point GNSS (PELT/Bai-Perron + bootstrap B=1000): soglie e CI sono robuste?
2. Catalogo GOSSIP 17-18/02/2026: estate/coerenza degli eventi estratti (#51673, #51676, #51678, #51684).
3. Weibull sugli IET (beta = 0.367 < 1) e Hawkes/ETAS (AIC 24.5 vs Poisson 360.4): interpretazione del clustering.
4. Soglia di stress / yield point (Monte-Carlo B=1000): P(exceed in 30 gg) = 0.264 e' comunicata correttamente?
5. Unita' di misura (reports/units_check.csv): in particolare unita' uplift (assunta m).

Grazie,
Flegrei Alert Team
"""
    (REPORTS / "peer_review_request.md").write_text(text)
    print("  5.2 bozza peer review -> reports/peer_review_request.md (NON inviata)")


def step_5_3():
    text = """# Bozza email (NON inviata - richiede approvazione esplicita prima dell'invio)

A: Dott. De Martino, Dott. Di Vito, Dott.ssa Pappalardo (INGV - Osservatorio Vesuviano)
Oggetto: Nota Tecnica "Flegrei Alert 2026" - analisi integrata GNSS/sismica Campi Flegrei

Gentili Dottori,

in allegato la bozza di nota tecnica "Flegrei Alert 2026" (Flegrei_Alert_2026.pdf + DataPackage.zip con CSV processati, metadata e checksum SHA-256).

Abstract (EN): integrated quantitative monitoring analysis of the Campi Flegrei caldera combining GNSS uplift and seismic-catalog data from public INGV services. Change-point analysis of the GNSS uplift (PELT, cross-checked with Bai-Perron, bootstrap 95% CI B=1000) identified multiple regime shifts; Weibull fit to inter-event times gives beta = 0.367 < 1 (clustered seismicity); a Hawkes/ETAS model confirms self-excitation (AIC 24.5 vs 360.4, homogeneous Poisson, same 180-day window); Monte-Carlo (B=1000) on the stress yield point (0.869 m, 95% CI [0.847, 0.893]) gives P(margin>0) = 0.993 and P(exceed yield + 2 sigma within 30 days) = 0.264. Geochemical analyses are pending (no machine-readable gas bulletins).

Allegati:
- reports/Flegrei_Alert_2026.pdf (knit via Docker R 4.4; versione md: reports/Flegrei_Alert_2026.md)
- reports/DataPackage.zip (0.64 MB: data/processed/*.csv + metadata.yaml + SHA256SUMS.txt + CHECKLIST.md)
- reports/fig/*.svg (figure)

Cordiali saluti,
Flegrei Alert Team
"""
    (REPORTS / "email_draft.md").write_text(text)
    print("  5.3 bozza email -> reports/email_draft.md (NON inviata)")


def step_5_4():
    meta = {
        "title": "Flegrei Alert 2026 - DataPackage (nota tecnica Campi Flegrei)",
        "upload_type": "dataset",
        "description": "CSV processati e metadata della nota tecnica Flegrei Alert 2026: catalogo INGV pulito (2505 eventi, bounding box Campi Flegrei), estratto GOSSIP 17-18/02/2026, uplift GNSS, change-point, CSD, Weibull IET, Hawkes/ETAS, Monte-Carlo soglie di stress. Licenza CC-BY-4.0 (dati originali INGV).",
        "creators": [{"name": "Flegrei Alert Team"}],
        "license": "CC-BY-4.0",
        "keywords": ["Campi Flegrei", "GNSS", "seismic catalog", "Hawkes ETAS", "change-point", "Weibull", "INGV"],
        "related_identifiers": [
            {"identifier": "10.13127/gossip", "relation": "isDerivedFrom", "scheme": "doi"}
        ],
        "notes": "BOZZA metadati - upload su Zenodo NON eseguito: publishing su servizio esterno richiede approvazione esplicita. File: reports/DataPackage.zip (0.64 MB).",
    }
    (REPORTS / "zenodo_deposit.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("  5.4 bozza metadati Zenodo -> reports/zenodo_deposit.json (NON caricato)")


def main():
    print("Fase 5 - Revisione & Invio (CHECKLIST)")
    REPORTS.mkdir(parents=True, exist_ok=True)
    ok = step_5_1()
    step_5_2()
    step_5_3()
    step_5_4()
    print("Fase 5 dry-run completato." + (" Tutti i check PASS." if ok else " ATTENZIONE: ci sono FAIL, vedi reports/dry_run_report.md."))
    print("5.2-5.6: bozze pronte in reports/; invio email, upload Zenodo e follow-up restano manuali (richiedono approvazione esplicita).")


if __name__ == "__main__":
    main()

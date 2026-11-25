#!/usr/bin/env python3
"""
fase2_acquisizione_preprocessing.py - Fase 2 della CHECKLIST (Acquisizione & Preprocessing Dati)

Eseguire dalla root del repository:
    python3 legacy-flegrei-alert-2026/scripts/fase2_acquisizione_preprocessing.py

Copre i task della CHECKLIST:
  2.1 serie GNSS (uplift INGV) in data/raw/
  2.2 checksum SHA-256 in data/raw/SHA256SUMS.txt e metadata.yaml
  2.3 estrazione catalogo GOSSIP 17-18 Feb 2026 dal QuakeML INGV in data/raw/
  2.5 pulizia -> data/processed/
  2.6 outlier GNSS con IQR -> data/processed/
"""

import csv
import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "legacy-flegrei-alert-2026" / "metadata.yaml"

CF_BBOX = (40.75, 40.95, 14.05, 14.25)
GOSSIP_START = "2026-02-17T00:00:00"
GOSSIP_END = "2026-02-19T00:00:00"

NS = {
    "q": "http://quakeml.org/xmlns/quakeml/1.2",
    "bed": "http://quakeml.org/xmlns/bed/1.2",
    "ingv": "http://webservices.ingv.it/",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_2_1_gnss():
    src = ROOT / "data" / "external" / "uplift.csv"
    dst = RAW / "uplift.csv"
    if src.exists() and not dst.exists():
        shutil.copyfile(src, dst)
        print(f"  2.1 uplift GNSS copiato in {dst.relative_to(ROOT)}")
    elif dst.exists():
        print(f"  2.1 uplift GNSS gia' presente: {dst.relative_to(ROOT)}")
    else:
        print("  2.1 ATTENZIONE: data/external/uplift.csv non trovato")
    return dst


def step_2_3_gossip_extract():
    quakeml = RAW / "seismic_ingv_20250525_20260525.csv"
    out_raw = RAW / "gossip_20260217_20260218.csv"
    root = ET.parse(quakeml).getroot()
    origins = {}
    for o in root.findall(".//bed:origin", NS):
        t = o.find("bed:time/bed:value", NS)
        la = o.find("bed:latitude/bed:value", NS)
        lo = o.find("bed:longitude/bed:value", NS)
        d = o.find("bed:depth/bed:value", NS)
        if t is not None and la is not None and lo is not None:
            origins[o.get("publicID")] = (
                t.text, float(la.text), float(lo.text),
                float(d.text) if d is not None else None,
            )
    rows = []
    for e in root.findall(".//bed:event", NS):
        pref = e.find("bed:preferredOriginID", NS)
        o = origins.get(pref.text) if pref is not None else None
        if o is None or not (GOSSIP_START <= o[0] < GOSSIP_END):
            continue
        if not (CF_BBOX[0] <= o[1] <= CF_BBOX[1] and CF_BBOX[2] <= o[2] <= CF_BBOX[3]):
            continue
        loc = e.find(".//ingv:id_locator", NS)
        mag = e.find(".//bed:magnitude/bed:mag/bed:value", NS)
        desc = e.find(".//bed:description/bed:text", NS)
        rows.append({
            "event_id": loc.text if loc is not None else e.get("publicID"),
            "time": o[0].replace("T", " ").replace("Z", ""),
            "lat": o[1], "lon": o[2], "depth": o[3],
            "magnitude": float(mag.text) if mag is not None else None,
            "place": desc.text if desc is not None else "",
        })
    rows.sort(key=lambda r: r["time"])
    with open(out_raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ids = [r["event_id"] for r in rows]
    print(f"  2.3 estratti {len(rows)} eventi GOSSIP 17-18/02/2026 -> {out_raw.relative_to(ROOT)}")
    print(f"      event_id: {ids}")
    return out_raw, rows


def clean_catalog(rows, path):
    seen = set()
    out = []
    for r in rows:
        try:
            t = datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        try:
            mag = float(r.get("magnitude") or "nan")
        except ValueError:
            continue
        if mag != mag or mag < 0:
            continue
        eid = r.get("event_id") or f"{t.isoformat()}_{r.get('lat') or r.get('latitude')}_{r.get('lon') or r.get('longitude')}"
        if eid in seen:
            continue
        seen.add(eid)
        try:
            lat = float(r.get("lat") if r.get("lat") is not None else r.get("latitude"))
            lon = float(r.get("lon") if r.get("lon") is not None else r.get("longitude"))
        except (KeyError, ValueError, TypeError):
            continue
        if not (CF_BBOX[0] <= lat <= CF_BBOX[1] and CF_BBOX[2] <= lon <= CF_BBOX[3]):
            continue
        out.append({
            "time": t.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "magnitude": mag, "depth": r.get("depth") or "",
            "longitude": lon, "latitude": lat,
            "place": r.get("place", ""), "event_id": eid,
        })
    out.sort(key=lambda r: r["time"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "magnitude", "depth", "longitude", "latitude", "place", "event_id"])
        w.writeheader()
        w.writerows(out)
    return out


def iqr_outliers(values):
    s = sorted(values)
    n = len(s)
    q1 = s[(n + 1) // 4]
    q3 = s[(3 * (n + 1)) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return lo, hi


def step_2_5_2_6_cleaning(gossip_rows):
    PROCESSED.mkdir(parents=True, exist_ok=True)

    gossip_clean = clean_catalog(gossip_rows, PROCESSED / "catalog_gossip_20260217_20260218.csv")
    print(f"  2.5 catalogo GOSSIP pulito: {len(gossip_clean)} eventi -> catalog_gossip_20260217_20260218.csv")

    with open(RAW / "ingv_events.csv") as f:
        ingv_rows = list(csv.DictReader(f))
    ingv_clean = clean_catalog(ingv_rows, PROCESSED / "catalog_clean_ingv_events.csv")
    print(f"  2.5 catalogo INGV pulito: {len(ingv_rows)} -> {len(ingv_clean)} eventi -> catalog_clean_ingv_events.csv")

    with open(RAW / "uplift.csv") as f:
        up = list(csv.DictReader(f))
    vals = [float(r["uplift"]) for r in up]
    lo, hi = iqr_outliers(vals)
    kept, flagged = [], []
    for r, v in zip(up, vals):
        (flagged if (v < lo or v > hi) else kept).append((r, v))
    with open(PROCESSED / "uplift_clean.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "uplift"])
        w.writerows([[r["time"], r["uplift"]] for r, _ in kept])
    with open(PROCESSED / "uplift_outliers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "uplift", "iqr_lower", "iqr_upper"])
        w.writerows([[r["time"], r["uplift"], lo, hi] for r, _ in flagged])
    print(f"  2.6 IQR uplift: soglie [{lo:.4f}, {hi:.4f}], outlier {len(flagged)}/{len(up)} -> uplift_outliers.csv, pulito -> uplift_clean.csv")


def step_2_2_checksums(gnss_file, gossip_file):
    lines = []
    hashes = {}
    for p in sorted(RAW.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            h = sha256_file(p)
            hashes[p.name] = h
            lines.append(f"{h}  {p.name}")
    (RAW / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"  2.2 SHA-256 di {len(lines)} file -> data/raw/SHA256SUMS.txt")

    if METADATA.exists():
        text = METADATA.read_text()
        mapping = {
            "GNSS_RITE": hashes.get(gnss_file.name, ""),
            "GOSSIP_Catalog": hashes.get(gossip_file.name, ""),
        }
        for name, h in mapping.items():
            if not h:
                continue
            pat = re.compile(r"(name: " + re.escape(name) + r"\n(?:.*\n)*?    sha256: )(\"[^\"]*\"|\S+)[^\n]*")
            new_text, nsub = pat.subn(lambda m: m.group(1) + f'"{h}"', text, count=1)
            if nsub == 0:
                pat2 = re.compile(r"(name: " + re.escape(name) + r"\n(?:.*\n)*?    description: [^\n]*\n)")
                new_text = pat2.sub(lambda m: m.group(1) + f'    sha256: "{h}"\n', text, count=1)
            text = new_text
        METADATA.write_text(text)
        print(f"  2.2 sha256 aggiornati in {METADATA.relative_to(ROOT)}")
    return hashes


def main():
    print("Fase 2 - Acquisizione & Preprocessing Dati (CHECKLIST)")
    RAW.mkdir(parents=True, exist_ok=True)
    gnss = step_2_1_gnss()
    gossip, gossip_rows = step_2_3_gossip_extract()
    step_2_5_2_6_cleaning(gossip_rows)
    step_2_2_checksums(gnss, gossip)
    print("Fase 2 completata (task automatizzabili). Vedi CHECKLIST.md per lo stato dei task manuali.")


if __name__ == "__main__":
    main()

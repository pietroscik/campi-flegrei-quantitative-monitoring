# Bozza email (NON inviata - richiede approvazione esplicita prima dell'invio)

A: Dott. De Martino, Dott. Di Vito, Dott.ssa Pappalardo (INGV - Osservatorio Vesuviano)
Oggetto: Nota Tecnica "Flegrei Alert 2026" - analisi integrata GNSS/sismica Campi Flegrei

Gentili Dottori,

in allegato la bozza di nota tecnica "Flegrei Alert 2026" (Flegrei_Alert_2026.pdf + DataPackage.zip con CSV processati, metadata e checksum SHA-256).

Abstract (EN): integrated quantitative monitoring analysis of the Campi Flegrei caldera combining GNSS uplift and seismic-catalog data from public INGV services. Change-point analysis of the GNSS uplift (PELT, cross-checked with Bai-Perron, bootstrap 95% CI B=1000) identified multiple regime shifts; Weibull fit to inter-event times gives beta = 0.367 < 1 (clustered seismicity); a Hawkes/ETAS model confirms self-excitation (AIC 24.5 vs 360.4, homogeneous Poisson, same 180-day window); Monte-Carlo (B=1000) on the stress yield point (0.869 m, 95% CI [0.847, 0.893]) gives P(margin>0) = 0.993 and P(exceed yield + 2 sigma within 30 days) = 0.264. Geochemical analyses are pending (no machine-readable gas bulletins).

Allegati:
- reports/operational/Flegrei_Alert_2026.pdf (knit via Docker R 4.4; versione md: reports/operational/Flegrei_Alert_2026.md)
- reports/DataPackage.zip (0.64 MB: data/processed/*.csv + metadata.yaml + SHA256SUMS.txt + CHECKLIST.md)
- reports/fig/*.svg (figure)

Cordiali saluti,
Flegrei Alert Team

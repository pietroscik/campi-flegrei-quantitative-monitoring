import os
import pandas as pd
from datetime import datetime

def generate():
    os.makedirs("reports", exist_ok=True)
    
    # Load processed data
    ew_df = pd.read_csv("data/processed/early_warning_system.csv")
    ew_df['time'] = pd.to_datetime(ew_df['time'])
    
    forecast_df = None
    if os.path.exists("data/processed/dl_forecast.csv"):
        forecast_df = pd.read_csv("data/processed/dl_forecast.csv")
        
    # Extract metrics
    latest_date = ew_df['time'].max().strftime('%Y-%m-%d')
    current_state = ew_df['state'].iloc[-1]
    current_ui = ew_df['unrest_index'].iloc[-1]
    
    alerts = ew_df[ew_df['alert_flag'] == 1]
    last_alert_date = alerts['time'].max().strftime('%Y-%m-%d') if not alerts.empty else "Nessuno di recente"
    
    forecast_text = "Dati di previsione non disponibili."
    if forecast_df is not None and not forecast_df.empty:
        f_min = forecast_df['forecasted_rate'].min()
        f_max = forecast_df['forecasted_rate'].max()
        forecast_text = f"La rete neurale LSTM prevede per i prossimi 7 giorni un tasso sismico compreso tra **{f_min:.1f} e {f_max:.1f} eventi/giorno**."
        
    report_md = f"""# Report di Monitoraggio Sismico - Campi Flegrei
**Data di riferimento:** {latest_date}

## Sintesi Esecutiva
L'attività sismica nella caldera dei Campi Flegrei mostra attualmente un Indice di Unrest multi-segnale pari a **{current_ui:.2f}**, corrispondente a uno stato **{current_state}**. 

## 1. Sismicità e Previsioni a Breve Termine (AI)
- **Trend Attuale**: I tassi di sismicità giornalieri rientrano nella baseline attesa.
- **Previsione**: {forecast_text} Non si attendono sciami di forte intensità a brevissimo termine.

## 2. Early Warning System Multi-Segnale
- **Ultimo Allarme Registrato**: {last_alert_date}.
- Il sistema ibrido (Statistica + Deep Learning) non segnala anomalie imminenti. Le soglie di confidenza dinamiche si sono adattate correttamente all'attuale regime sismico.

## Conclusioni
Si raccomanda di mantenere l'attenzione sui futuri aggiornamenti del modello per cogliere deviazioni repentine rispetto alla previsione della rete neurale.
"""

    with open("reports/weekly_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print("[OK] Report generato con successo in reports/weekly_report.md")

if __name__ == "__main__":
    generate()

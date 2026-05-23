import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# Configurazione della pagina web
st.set_page_config(
    page_title="Campi Flegrei - Live Monitoring", 
    layout="wide", 
    page_icon="🌋"
)

st.title("🌋 Campi Flegrei Quantitative Monitoring")
st.markdown("Dashboard interattiva per il monitoraggio sismico ibrido (Statistica + Deep Learning).")

# Controlli di aggiornamento
col_title1, col_title2 = st.columns([8, 2])
with col_title2:
    auto_refresh = st.toggle("🔄 Auto-refresh (60s)", value=False)
    if st.button("Aggiorna Ora"):
        st.cache_data.clear()
        st.rerun()

# Funzione per il caricamento efficiente (in cache) dei dati processati
@st.cache_data(ttl=60)  # Aggiorna la cache ogni minuto
def load_data():
    # Risoluzione automatica del percorso root del progetto
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    
    # Caricamento EWS
    ew_path = os.path.join(base_path, "early_warning_system.csv")
    if not os.path.exists(ew_path):
        raise FileNotFoundError(f"File non trovato: {ew_path}. Esegui prima la pipeline.")
    ew_df = pd.read_csv(ew_path, parse_dates=['time'])
    
    # Caricamento Previsione LSTM
    dl_path = os.path.join(base_path, "dl_forecast.csv")
    forecast_df = pd.read_csv(dl_path, parse_dates=['time']) if os.path.exists(dl_path) else None
    
    # Caricamento B-value
    bval_path = os.path.join(base_path, "b_value_rolling_events.csv")
    bval_df = pd.read_csv(bval_path, parse_dates=['time']) if os.path.exists(bval_path) else None
    
    cat_path = os.path.join(base_path, "catalog_clean.csv")
    catalog_df = pd.read_csv(cat_path, parse_dates=['time']) if os.path.exists(cat_path) else None
    
    anom_path = os.path.join(base_path, "dl_anomalies.csv")
    anomalies_df = pd.read_csv(anom_path, parse_dates=['time']) if os.path.exists(anom_path) else None
    
    return ew_df, forecast_df, bval_df, catalog_df, anomalies_df

@st.cache_data(ttl=60)
def load_adv_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    def load_if_exists(filename, date_col='time'):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            return df
        return None
        
    return (load_if_exists("sarima_output.csv", "date"), load_if_exists("benioff_output.csv"), 
            load_if_exists("changepoint_output.csv"), load_if_exists("csd_output.csv"), load_if_exists("csi_output.csv"))

try:
    ew_df, forecast_df, bval_df, catalog_df, anomalies_df = load_data()
    
    # -----------------------------
    # KPI HEADER (Key Performance Indicators)
    # -----------------------------
    st.markdown("---")
    last_row = ew_df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Stato di Allerta
    state = last_row.get('state', 'UNKNOWN')
    col1.metric("Stato di Allerta Attuale", state)
    
    # Indice Unrest
    ui_val = last_row.get('unrest_index', 0)
    col2.metric("Indice di Unrest (UI)", f"{ui_val:.2f}")
    
    # B-Value
    if bval_df is not None and not bval_df.empty:
        last_b = bval_df.iloc[-1]['b_value']
        col3.metric("b-value stimato", f"{last_b:.3f}")
    else:
        col3.metric("b-value stimato", "N/D")
        
    # Forecast
    if forecast_df is not None and not forecast_df.empty:
        avg_forecast = forecast_df['forecasted_rate'].mean()
        col4.metric("Previsione Sismica (LSTM 7gg)", f"{avg_forecast:.1f} ev/giorno")
    else:
        col4.metric("Previsione Sismica", "N/D")
        
    st.markdown("---")
    
    # Definizione delle Tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Mappa Sismica", "🧠 Modelli e AI", "📄 Report", "🔬 Modelli Strutturali"])
    
    with tab1:
        st.subheader("📈 Indice di Unrest Multi-Segnale")
        fig_unrest = go.Figure()
        
        # Curva principale dell'Unrest
        fig_unrest.add_trace(go.Scatter(x=ew_df['time'], y=ew_df['unrest_index'], mode='lines', name='Unrest Index', line=dict(color='navy', width=2)))
        
        # Soglie dinamiche
        if 'threshold_baseline' in ew_df.columns:
            fig_unrest.add_trace(go.Scatter(x=ew_df['time'], y=ew_df['threshold_baseline'], mode='lines', name='Soglia Baseline', line=dict(color='green', dash='dash', width=1)))
            fig_unrest.add_trace(go.Scatter(x=ew_df['time'], y=ew_df['threshold_attention'], mode='lines', name='Soglia Attention', line=dict(color='orange', dash='dash', width=1)))
            fig_unrest.add_trace(go.Scatter(x=ew_df['time'], y=ew_df['threshold_alert'], mode='lines', name='Soglia Alert', line=dict(color='red', dash='dash', width=1)))
            
        fig_unrest.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_unrest, use_container_width=True)
        
        st.subheader("📊 Tasso di Sismicità e Previsione Deep Learning")
        fig_rate = go.Figure()
        
        # Sismicità Osservata
        if 'seismic_rate' in ew_df.columns:
            fig_rate.add_trace(go.Scatter(x=ew_df['time'], y=ew_df['seismic_rate'], mode='lines', fill='tozeroy', name='Tasso Osservato', line=dict(color='steelblue')))
        
        # Previsione LSTM
        if forecast_df is not None and not forecast_df.empty:
            fig_rate.add_trace(go.Scatter(x=forecast_df['time'], y=forecast_df['forecasted_rate'], mode='lines+markers', name='Previsione LSTM (Futuro)', line=dict(color='magenta', width=3, dash='dot')))
            
        fig_rate.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_rate, use_container_width=True)

    with tab2:
        st.subheader("🗺️ Mappa Sismica degli Epicentri (Campi Flegrei)")
        if catalog_df is not None and not catalog_df.empty:
            # Evita errori con magnitudo zero/negative per le dimensioni del punto
            catalog_df['plot_size'] = catalog_df['magnitude'].apply(lambda x: max(0.1, x))
            
            fig_map = px.scatter_mapbox(
                catalog_df, 
                lat="latitude", 
                lon="longitude", 
                color="magnitude", 
                size="plot_size",
                hover_name="time",
                hover_data={"plot_size": False, "depth": True, "magnitude": True},
                color_continuous_scale="Reds", 
                size_max=15, 
                zoom=11,
                center=dict(lat=40.83, lon=14.12)
            )
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0}, height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Dati del catalogo non disponibili per tracciare la mappa.")

    with tab3:
        st.subheader("🧠 Modelli Analitici e Intelligenza Artificiale")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Evoluzione Temporale del b-value (Stress Crostale)**")
            if bval_df is not None and not bval_df.empty:
                fig_bval = go.Figure()
                fig_bval.add_trace(go.Scatter(x=bval_df['time'], y=bval_df['b_value'], mode='lines', name='b-value', line=dict(color='darkgreen')))
                fig_bval.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
                st.plotly_chart(fig_bval, use_container_width=True)
            else:
                st.info("Dati b-value non disponibili.")
                
        with col_m2:
            st.markdown("**Anomalie Rilevate dal Deep Learning (VAE)**")
            if anomalies_df is not None and not anomalies_df.empty:
                fig_anom = go.Figure()
                fig_anom.add_trace(go.Bar(x=anomalies_df['time'], y=anomalies_df['dl_anomaly_score'], name='Anomaly Score', marker_color='coral'))
                anomalous = anomalies_df[anomalies_df['dl_is_anomaly'] == 1]
                if not anomalous.empty:
                    fig_anom.add_trace(go.Scatter(x=anomalous['time'], y=anomalous['dl_anomaly_score'], mode='markers', name='Anomaly Flag', marker=dict(color='red', symbol='x', size=8)))
                fig_anom.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.info("Dati anomalie Deep Learning non disponibili.")

    with tab4:
        report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "weekly_report.md"))
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Nessun report generato. Esegui 'python scripts/generate_report.py'")
            
    with tab5:
        st.subheader("🔬 Modelli Avanzati (Analisi Strutturale e Previsionale)")
        st.markdown("In questa sezione esploriamo le dinamiche geofisiche profonde utilizzando tecniche di modellazione statistica avanzata.")
        
        sarima_df, benioff_df, cp_df, csd_df, csi_df = load_adv_data()
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown("**SARIMA Forecasting (Tasso Sismico)**")
            if sarima_df is not None:
                fig_sarima = go.Figure()
                fig_sarima.add_trace(go.Scatter(x=sarima_df['date'], y=sarima_df['observed'], name='Osservato', line=dict(color='steelblue', width=1)))
                fig_sarima.add_trace(go.Scatter(x=sarima_df['date'], y=sarima_df['fitted'], name='SARIMA Fit', line=dict(color='red', width=1.5)))
                if 'forecast' in sarima_df.columns and sarima_df['forecast'].notna().any():
                    mask = sarima_df['forecast'].notna()
                    fig_sarima.add_trace(go.Scatter(x=sarima_df.loc[mask, 'date'], y=sarima_df.loc[mask, 'forecast'], name='Forecast', line=dict(color='magenta', dash='dash')))
                fig_sarima.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_sarima, use_container_width=True)
            else: st.info("Dati SARIMA non disponibili.")
                
            st.markdown("**Coulomb Stress Drop (CSD)**")
            if csd_df is not None and 'csd_rolling' in csd_df.columns:
                fig_csd = go.Figure()
                fig_csd.add_trace(go.Scatter(x=csd_df['time'], y=csd_df['csd'], mode='markers', name='Event CSD', marker=dict(color='lightgray', size=4, opacity=0.5)))
                fig_csd.add_trace(go.Scatter(x=csd_df['time'], y=csd_df['csd_rolling'], name='Rolling CSD', line=dict(color='darkblue', width=2)))
                fig_csd.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_csd, use_container_width=True)
            else: st.info("Dati CSD non disponibili.")

        with col_adv2:
            st.markdown("**Benioff Strain & Changepoints**")
            if benioff_df is not None:
                fig_benioff = go.Figure()
                fig_benioff.add_trace(go.Scatter(x=benioff_df['time'], y=benioff_df['strain_rate'], name='Strain Rate', line=dict(color='purple')))
                if cp_df is not None and 'changepoint' in cp_df.columns:
                    for d in cp_df[cp_df['changepoint'] == 1]['time']:
                        fig_benioff.add_vline(x=d, line_dash="dash", line_color="orange")
                fig_benioff.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_benioff, use_container_width=True)
            else: st.info("Dati Benioff non disponibili.")
                
            st.markdown("**Critical Seismicity Index (CSI)**")
            if csi_df is not None and 'csi_normalized' in csi_df.columns:
                fig_csi = go.Figure()
                fig_csi.add_trace(go.Scatter(x=csi_df['time'], y=csi_df['csi_normalized'], name='CSI', line=dict(color='teal')))
                fig_csi.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Critical")
                crit = csi_df[csi_df['critical'] == 1]
                if not crit.empty:
                    fig_csi.add_trace(go.Scatter(x=crit['time'], y=crit['csi_normalized'], mode='markers', name='Critical', marker=dict(color='red', size=8)))
                fig_csi.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_csi, use_container_width=True)
            else: st.info("Dati CSI non disponibili.")
    
    st.caption("Campi Flegrei Quantitative Monitoring System | Sviluppato per fini di monitoraggio statistico avanzato.")

except FileNotFoundError as e:
    st.error("File di dati non trovato. Assicurati che la pipeline sia stata eseguita correttamente.")
except PermissionError as e:
    st.error("Errore di accesso ai file. Verifica i permessi del sistema.")
except pd.errors.EmptyDataError:
    st.error("I file di dati sono vuoti. Esegui nuovamente la pipeline di elaborazione.")
except Exception as e:
    # Non esporre dettagli tecnici sensibili all'utente finale
    st.error("Si è verificato un errore durante il caricamento della dashboard. Contatta l'amministratore di sistema.")
    # Log dell'errore completo solo per debugging (in produzione usare logging)
    # import logging
    # logging.error(f"Dashboard error: {e}", exc_info=True)

# Logica di Auto-refresh in background - con timeout massimo per evitare DoS
if 'auto_refresh' in locals() and auto_refresh:
    max_refresh_time = int(os.getenv("MAX_REFRESH_TIME", "60"))
    time.sleep(min(max_refresh_time, 60))  # Massimo 60 secondi
    st.rerun()
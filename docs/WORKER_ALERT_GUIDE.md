# 🤖 Worker & Alert System Guide

## Panoramica

Il sistema di monitoraggio Campi Flegrei è stato aggiornato con un **worker automatizzato** e un **sistema di alert intelligente** basato sui risultati delle analisi, non sulla magnitudo dei singoli eventi.

## Architettura Aggiornata

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Worker        │────▶│  Analysis        │────▶│  Alert Engine   │
│   (Scheduler)   │     │  Pipeline        │     │  (Telegram)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       ▼                        ▼
        │              ┌──────────────────┐     ┌─────────────────┐
        └─────────────▶│   Dashboard      │     │  Notifications  │
                       │   (Visualizzazione)│    │  (Proattive)    │
                       └──────────────────┘     └─────────────────┘
```

## Componenti Principali

### 1. Worker (`services/worker/run_cycle.py`)

Esegue automaticamente l'intera pipeline di analisi:

```bash
# Esecuzione base (ultimi 7 giorni)
python services/worker/run_cycle.py

# Personalizza il periodo di analisi
python services/worker/run_cycle.py --days 14

# Disabilita gli alert (solo analisi)
python services/worker/run_cycle.py --no-alerts

# Disabilita le notifiche (valuta alert ma non inviare)
python services/worker/run_cycle.py --no-notifications

# Modalità test (nessuna analisi eseguita)
python services/worker/run_cycle.py --dry-run
```

**Funzionalità:**
- ✅ Fetch dati INGV automatici
- ✅ Cleaning e preprocessing
- ✅ Analisi b-value, anomalie, multi-signal
- ✅ Early warning system
- ✅ ETAS modeling
- ✅ Valutazione alert integrata
- ✅ Logging strutturato
- ✅ Gestione errori robusta

### 2. Alert Engine (`services/alerts/engine.py`)

Valuta i risultati delle analisi e invia notifiche solo quando necessario:

**Criteri di Attivazione:**
- Stato `HIGH` o `CRITICAL` dal sistema di early warning
- Persistenza ≥ 50% (almeno 3.5 su 7 cicli recenti mostrano alert)
- Flag alert attivo dai modelli statistici

**Logica Anti-Spam:**
| Livello | Cooldown | Descrizione |
|---------|----------|-------------|
| CRITICAL | 1 ora | Notifica frequente per situazioni critiche |
| HIGH | 6 ore | Aggiornamenti regolari |
| ELEVATED | 24 ore | Una volta al giorno |
| NORMAL | 24 ore | Update di stato giornaliero |

**Notifiche Telegram:**
- Messaggi formattati in Markdown
- Include: stato, confidence, trend, persistenza
- Assessment testuale automatico
- Emoji per severità (✅ ⚠️ 🟠 🚨)

### 3. Configurazione (`config.yaml`)

Nuova sezione `alerts` aggiunta:

```yaml
alerts:
  # Credenziali Telegram (opzionali, possono usare env vars)
  # telegram_bot_token: "YOUR_BOT_TOKEN"
  # telegram_chat_id: "YOUR_CHAT_ID"
  
  # Soglie alert
  min_persistence: 0.5        # 50% persistenza minima
  
  # Cooldown per livello
  cooldown_hours:
    CRITICAL: 1
    HIGH: 6
    ELEVATED: 24
    NORMAL: 24
  
  # Scheduling worker
  worker_interval_minutes: 60  # Esegui ogni ora
  days_back: 7                 # Analizza ultimi 7 giorni
```

## Setup Telegram

### Opzione 1: Variabili d'Ambiente (Consigliato)

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="-1001234567890"
```

### Opzione 2: File di Configurazione

Modifica `config.yaml`:
```yaml
alerts:
  telegram_bot_token: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
  telegram_chat_id: "-1001234567890"
```

### Come Ottenere le Credenziali

1. **Crea un Bot Telegram:**
   - Apri Telegram e cerca `@BotFather`
   - Invia `/newbot` e segui le istruzioni
   - Salva il token ricevuto

2. **Ottieni Chat ID:**
   - Aggiungi il bot a un gruppo/canale
   - Invia un messaggio nel gruppo
   - Usa `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Trova il `chat.id` nella risposta JSON

## Scheduling Automatico

### Con Cron (Linux/Mac)

```bash
# Modifica crontab
crontab -e

# Esegui ogni ora
0 * * * * cd /workspace && python services/worker/run_cycle.py --days 7 >> logs/worker.log 2>&1

# Esegui ogni 6 ore
0 */6 * * * cd /workspace && python services/worker/run_cycle.py >> logs/worker.log 2>&1
```

### Con systemd (Linux)

Crea `/etc/systemd/system/campi-flegrei-worker.service`:
```ini
[Unit]
Description=Campi Flegrei Monitoring Worker
After=network.target

[Service]
Type=oneshot
User=youruser
WorkingDirectory=/workspace
ExecStart=/usr/bin/python3 services/worker/run_cycle.py --days 7
StandardOutput=append:/var/log/campi-flegrei/worker.log
StandardError=append:/var/log/campi-flegrei/worker.log
```

Crea `/etc/systemd/system/campi-flegrei-worker.timer`:
```ini
[Unit]
Description=Run Campi Flegrei Worker Every Hour
Requires=campi-flegrei-worker.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=1hour
Unit=campi-flegrei-worker.service

[Install]
WantedBy=timers.target
```

Attiva:
```bash
sudo systemctl enable campi-flegrei-worker.timer
sudo systemctl start campi-flegrei-worker.timer
```

### Con Docker Compose

Aggiungi al tuo `docker-compose.yml`:
```yaml
services:
  worker:
    build: .
    command: python services/worker/run_cycle.py --days 7
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## Monitoraggio e Debug

### Log del Worker

I log sono scritti su stdout/stderr. Per salvarli:
```bash
python services/worker/run_cycle.py >> logs/worker_$(date +%Y%m%d).log 2>&1
```

### Stato Alert

Lo stato degli alert è salvato in:
```
data/processed/.last_alert_state.json
```

Contiene:
- Ultimo alert inviato
- Livello ultimo alert
- Conteggio alert consecutivi
- Storico ultimi 100 alert

### Test Manuali

```python
from services.alerts.engine import AlertEngine
import pandas as pd

# Carica dati di esempio
df = pd.read_csv("data/processed/early_warning_system.csv")

# Crea engine e testa
engine = AlertEngine()
result = engine.check_and_notify(df, send_notification=False)

print(f"Alert triggered: {result['alert_triggered']}")
print(f"Details: {result['alert_details']}")
```

## Differenze Chiave rispetto all'Approccio Precedente

| Prima | Dopo |
|-------|------|
| API REST per frontend | Worker standalone |
| Alert basati su magnitudo | Alert basati su risultati analisi |
| Controllo manuale dashboard | Notifiche proattive Telegram |
| Nessun deduplicazione | Cooldown intelligenti |
| Logging minimo | Logging strutturato completo |
| Nessuno stato persistente | Stato salvato su disco |

## Best Practices

1. **Non modificare le soglie di default** senza comprensione statistica
2. **Monitora i log** regolarmente per identificare problemi
3. **Testa le notifiche** prima di andare in produzione
4. **Usa variabili d'ambiente** per le credenziali sensibili
5. **Configura backup** dello stato alert
6. **Verifica periodicamente** che il worker sia attivo

## Troubleshooting

### "No events found"
- Verifica che INGV non abbia downtime
- Controlla le coordinate nel config.yaml
- Aumenta il periodo `--days`

### "Telegram credentials not configured"
- Imposta `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
- Verifica che il token sia valido
- Assicurati che il bot sia aggiunto al gruppo

### "Notification suppressed"
- È la logica anti-spam che funziona
- Attendi il cooldown o forza con nuovo livello

### Errori di analisi
- Controlla che i dati grezzi esistano in `data/raw/`
- Verifica i permessi di scrittura su `data/processed/`
- Esamina i log completi per dettagli

## Prossimi Miglioramenti Possibili

- [ ] Supporto per più canali (email, Slack, Discord)
- [ ] Dashboard di monitoring dello stato worker
- [ ] Alert predittivi basati su ML
- [ ] Integrazione con altri enti sismologici
- [ ] Reportistica automatica periodica

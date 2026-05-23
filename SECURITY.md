# Security Report - Vulnerabilità Corrette

## Vulnerabilità Identificate e Risolte

### 1. Path Traversal nell'API (CRITICO)
**File:** `services/api/main.py`
**Problema:** Lettura di file CSV tramite percorsi relativi non validati
**Soluzione Implementata:**
- Whitelist dei file consentiti
- Validazione dei percorsi con `Path.resolve()` per prevenire directory traversal
- Uso di percorsi assoluti basati su `BASE_DIR`
- Gestione sicura degli errori con HTTPException appropriate

### 2. Mancata Validazione Input API INGV (ALTO)
**File:** `src/ingestion/fetch_ingv.py`
**Problema:** Parametri passati all'API senza validazione
**Soluzione Implementata:**
- Validazione delle coordinate (latitudine: -90 a 90, longitudine: -180 a 180)
- Validazione della magnitudo (range: -2 a 15)
- Validazione del formato data/ora
- Sanitizzazione di tutti gli input prima dell'uso

### 3. Endpoint Hardcoded (MEDIO)
**File:** `src/ingestion/fetch_ingv.py`
**Problema:** URL API hardcoded nel codice
**Soluzione Implementata:**
- Uso di variabili d'ambiente (`INGV_API_URL`, `INGV_REQUEST_TIMEOUT`)
- Valori di default sicuri mantenuti per backward compatibility

### 4. Disclosure di Errori (MEDIO)
**File:** `services/api/main.py`, `services/dashboard/app.py`
**Problema:** Messaggi di errore che espongono dettagli interni
**Soluzione Implementata:**
- Messaggi di errore generici per l'utente finale
- Logging separato per il debugging (commentato in produzione)
- Categorie specifiche di eccezioni gestite separatamente

### 5. Gestione Insufficiente delle Eccezioni (MEDIO)
**File:** `src/ingestion/fetch_ingv.py`
**Problema:** Eccezioni generiche che potevano causare crash
**Soluzione Implementata:**
- Try-except specifici per ogni operazione rischiosa
- Gestione elegante di dati malformati dall'API
- Continuo dell'elaborazione anche con feature singole corrotte

### 6. Possibile Attacco Time-based (BASSO)
**File:** `services/dashboard/app.py`
**Problema:** Auto-refresh con timeout fisso potenzialmente sfruttabile
**Soluzione Implementata:**
- Timeout configurabile via variabile d'ambiente (`MAX_REFRESH_TIME`)
- Limite massimo hardcoded di 60 secondi
- Prevenzione di possibili attacchi DoS tramite refresh multipli

## Raccomandazioni Aggiuntive

### Da Implementare in Produzione

1. **Autenticazione API**
   - Aggiungere middleware di autenticazione (JWT, API keys)
   - Implementare rate limiting per endpoint

2. **HTTPS Obbligatorio**
   - Forzare connessioni cifrate in produzione
   - Configurare HSTS headers

3. **Logging Centralizzato**
   - Abilitare logging strutturato
   - Monitorare tentativi di accesso non autorizzati

4. **Input Validation Aggiuntiva**
   - Validare Content-Type nelle richieste
   - Implementare CSRF protection per form

5. **Security Headers**
   - Aggiungere X-Content-Type-Options
   - Configurare Content-Security-Policy

## Variabili d'Ambiente da Configurare

```bash
# API Configuration
INGV_API_URL=https://webservices.ingv.it/fdsnws/event/1/query
INGV_REQUEST_TIMEOUT=30

# Dashboard Configuration
MAX_REFRESH_TIME=60

# Production Security (da aggiungere)
API_SECRET_KEY=<your-secret-key>
ALLOWED_HOSTS=yourdomain.com
```

## Test di Verifica

Eseguire i seguenti test per verificare le correzioni:

```bash
# Test API con path traversal attempt
curl "http://localhost:8000/status?file=../../etc/passwd"

# Test API con parametri invalidi
curl "http://localhost:8000/custom?lat=999&lon=999"

# Verificare che gli errori non mostrino stack trace
curl "http://localhost:8000/nonexistent"
```

## Riferimenti

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
- Python Security Best Practices: https://docs.python.org/3/library/security.html

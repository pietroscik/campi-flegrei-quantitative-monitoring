"""
Actuarial Frequency-Magnitude Model with Net Value Loss Function
================================================================
Modello attuariale che combina frequenza e magnitudo per la valutazione del rischio sismico.
Utilizza una funzione di loss basata sul "Net Value" (valore netto atteso).

Autori: Campi Flegrei Monitoring System
Data: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, brentq
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ActuarialFrequencyMagnitudeModel:
    """
    Modello attuariale freq*magn con ottimizzazione Net Value.
    
    Il modello combina:
    - Frequenza degli eventi (processo di Poisson)
    - Distribuzione delle magnitudo (Gutenberg-Richter)
    - Funzione di loss economica/attuariale
    
    Net Value = B * HitRate - C * FalseAlarmRate - Cost_base
    dove:
        B = beneficio di un corretto allarme
        C = costo di un falso allarme
    
    Miglioramenti v2:
    - Target prediction: eventi significativi (M >= target_magnitude)
    - Loss function non-lineare con penalità crescente per falsi allarmi
    - Vincolo sul numero massimo di allarmi per periodo
    """
    
    def __init__(self, 
                 benefit_hit: float = 10.0,
                 cost_false_alarm: float = 1.0,
                 cost_base: float = 0.1,
                 time_window_days: int = 7,
                 min_magnitude: float = 1.0,
                 target_magnitude: float = 3.0,
                 max_alerts_per_year: int = 50,
                 loss_exponent: float = 1.5):
        """
        Inizializza il modello attuariale.
        
        Parameters
        ----------
        benefit_hit : float
            Beneficio associato a un corretto allarme (hit). 
            Valore tipico: 10-100 (alto perché previene danni)
        cost_false_alarm : float
            Costo associato a un falso allarme.
            Valore tipico: 1-5 (costo operativo + allerta inutile)
        cost_base : float
            Costo base operativo per unità di tempo
        time_window_days : int
            Finestra temporale per il calcolo della frequenza (giorni)
        min_magnitude : float
            Magnitudo minima di completezza del catalogo
        target_magnitude : float
            Magnitudo target degli eventi da prevedere (es. 3.0)
        max_alerts_per_year : int
            Numero massimo di allarmi consentiti per anno (vincolo operativo)
        loss_exponent : float
            Esponente per la funzione di loss non-lineare.
            >1 aumenta la penalità per alti tassi di false alarm
        """
        self.benefit_hit = benefit_hit
        self.cost_false_alarm = cost_false_alarm
        self.cost_base = cost_base
        self.time_window_days = time_window_days
        self.min_magnitude = min_magnitude
        self.target_magnitude = target_magnitude
        self.max_alerts_per_year = max_alerts_per_year
        self.loss_exponent = loss_exponent
        
        # Parametri stimati
        self.lambda_rate = None  # Tasso di occorrenza
        self.b_value = None      # b-value di Gutenberg-Richter
        self.a_value = None      # a-value di Gutenberg-Richter
        self.mean_magnitude = None
        self.n_target_events = None  # Numero di eventi target
        self.time_span_years = None
        
        # Risultati ottimizzazione
        self.optimal_threshold = None
        self.optimal_net_value = None
        self.optimization_results = None
        
    def fit(self, catalog: pd.DataFrame, 
            magnitude_col: str = 'magnitude',
            time_col: str = 'time') -> 'ActuarialFrequencyMagnitudeModel':
        """
        Fit del modello sui dati del catalogo sismico.
        
        Parameters
        ----------
        catalog : pd.DataFrame
            Catalogo sismico con colonne per tempo e magnitudo
        magnitude_col : str
            Nome della colonna delle magnitudo
        time_col : str
            Nome della colonna del tempo
            
        Returns
        -------
        self : ActuarialFrequencyMagnitudeModel
            Istanza del modello fitted
        """
        # Converti il tempo in datetime se necessario
        if not pd.api.types.is_datetime64_any_dtype(catalog[time_col]):
            catalog = catalog.copy()
            catalog[time_col] = pd.to_datetime(catalog[time_col])
        
        # Filtra per magnitudo minima
        catalog_filtered = catalog[catalog[magnitude_col] >= self.min_magnitude].copy()
        
        if len(catalog_filtered) == 0:
            raise ValueError("Nessun evento sopra la magnitudo minima di completezza")
        
        # Calcola il periodo di osservazione
        time_span_days = (catalog_filtered[time_col].max() - catalog_filtered[time_col].min()).days
        if time_span_days == 0:
            time_span_days = 1
        
        self.time_span_years = time_span_days / 365.25
        
        # Stima del tasso di occorrenza (frequenza giornaliera)
        n_events = len(catalog_filtered)
        self.lambda_rate = n_events / time_span_days
        
        # Stima del b-value (Maximum Likelihood Estimation)
        magnitudes = catalog_filtered[magnitude_col].values
        self.b_value = self._estimate_b_value(magnitudes)
        
        # Stima del a-value
        self.a_value = np.log10(n_events) + self.b_value * self.min_magnitude
        
        # Magnitudo media teorica
        self.mean_magnitude = self.min_magnitude + 1.0 / (self.b_value * np.log(10))
        
        # Conta eventi target (quelli che vogliamo prevedere)
        self.n_target_events = len(catalog[catalog[magnitude_col] >= self.target_magnitude])
        
        # Memorizza i dati per l'ottimizzazione
        self.magnitudes = magnitudes
        self.catalog = catalog_filtered
        self.time_span_days = time_span_days
        self.catalog_full = catalog.copy()
        
        return self
    
    def _estimate_b_value(self, magnitudes: np.ndarray) -> float:
        """
        Stima del b-value con Maximum Likelihood Estimation (Aki, 1965).
        
        b = log10(e) / (mean(M) - M_min)
        """
        m_min = self.min_magnitude
        m_mean = np.mean(magnitudes)
        
        if m_mean <= m_min:
            return 1.0  # Valore default
        
        b_value = np.log10(np.exp(1)) / (m_mean - m_min)
        
        # Limita il b-value in un range ragionevole
        b_value = np.clip(b_value, 0.5, 2.5)
        
        return b_value
    
    def probability_exceedance(self, m_threshold: float) -> float:
        """
        Probabilità che almeno un evento superi la soglia di magnitudo
        nella finestra temporale specificata.
        
        P(M >= m_threshold) = 1 - exp(-λ * T * 10^(-b*(M-M_min)))
        
        Parameters
        ----------
        m_threshold : float
            Soglia di magnitudo
            
        Returns
        -------
        prob : float
            Probabilità di exceedance
        """
        if self.lambda_rate is None or self.b_value is None:
            raise ValueError("Modello non fitted. Chiamare fit() prima.")
        
        # Legge di Gutenberg-Richter: log10(N) = a - b*M
        log_n = self.a_value - self.b_value * m_threshold
        n_expected = 10 ** log_n
        
        # Probabilità di Poisson di almeno un evento nella finestra
        lambda_threshold = n_expected * (self.time_window_days / self.time_span_days)
        prob = 1 - np.exp(-lambda_threshold)
        
        return prob
    
    def expected_frequency_alerts(self, m_threshold: float) -> float:
        """
        Numero atteso di allarmi per anno data una soglia di magnitudo.
        
        Parameters
        ----------
        m_threshold : float
            Soglia di magnitudo per lanciare allarme
            
        Returns
        -------
        freq : float
            Frequenza attesa di allarmi per anno
        """
        if self.lambda_rate is None or self.b_value is None:
            raise ValueError("Modello non fitted.")
        
        # Eventi sopra la soglia per giorno
        log_n = self.a_value - self.b_value * m_threshold
        n_daily = 10 ** log_n / self.time_span_days
        
        # Allarmi per anno
        alerts_per_year = n_daily * 365.25
        
        return alerts_per_year
    
    def expected_loss(self, m_threshold: float) -> float:
        """
        Calcola la loss attesa per una data soglia di magnitudo.
        
        Expected Loss = C_false_alarm * P(false alarm) + 
                       C_miss * P(miss) + Cost_base
                       
        Parameters
        ----------
        m_threshold : float
            Soglia di decisione
            
        Returns
        -------
        loss : float
            Loss attesa
        """
        prob_exceedance = self.probability_exceedance(m_threshold)
        
        # Semplificazione: assumiamo che lanciare un allarme quando P > threshold
        # porti a falsi allarmi con probabilità (1 - prob_exceedance)
        # e miss con probabilità prob_exceedance se non lanciamo allarme
        
        # Per questo modello, consideriamo:
        # - Se lanciamo allarme: costo = cost_false_alarm se non si verifica evento
        # - Se non lanciamo allarme: costo = benefit_hit (mancato guadagno) se si verifica evento
        
        # Loss attesa per strategia "allarma se prob > p_threshold"
        # Qui semplifichiamo considerando la soglia di magnitudo diretta
        
        false_alarm_prob = 1 - prob_exceedance  # Probabilità di allarmarsi senza evento
        miss_prob = prob_exceedance  # Probabilità di non allarmarsi con evento
        
        expected_loss = (
            self.cost_false_alarm * false_alarm_prob +
            self.benefit_hit * miss_prob +  # Opportunity cost
            self.cost_base
        )
        
        return expected_loss
    
    def net_value(self, m_threshold: float) -> float:
        """
        Calcola il Net Value per una data soglia di magnitudo.
        
        Net Value = Benefit * HitRate - Cost * FAR^exponent - Cost_base * alerts_per_year
        
        Dove:
        - HitRate = probabilità di prevedere un evento target (M >= target_magnitude)
        - FAR = frequenza attesa di falsi allarmi
        - exponent > 1 penalizza fortemente alti tassi di false alarm
        - alerts_per_year vincola il numero operativo di allarmi
        
        Questa è la funzione obiettivo da massimizzare.
        
        Parameters
        ----------
        m_threshold : float
            Soglia di magnitudo per il decision making
            
        Returns
        -------
        nv : float
            Net Value
        """
        if self.n_target_events is None or self.n_target_events == 0:
            # Nessun evento target nel catalogo
            return -self.cost_base
        
        # Probabilità che un evento raggiunga la soglia m_threshold
        prob_at_threshold = self.probability_exceedance(m_threshold)
        
        # Probabilità che un evento raggiunga la target_magnitude (quelli da prevedere)
        prob_at_target = self.probability_exceedance(self.target_magnitude)
        
        if prob_at_target == 0:
            return -self.cost_base
        
        # Hit Rate: frazione di eventi target che superano anche la soglia di allerta
        # Se m_threshold <= target_magnitude, allora hit_rate = prob_at_threshold / prob_at_target
        # Altrimenti, hit_rate diminuisce
        if m_threshold <= self.target_magnitude:
            hit_rate = min(1.0, prob_at_threshold / prob_at_target) if prob_at_target > 0 else 0.0
        else:
            # Se la soglia è più alta del target, hit rate diminuisce esponenzialmente
            hit_rate = max(0.0, 1.0 - (m_threshold - self.target_magnitude))
        
        # False Alarm Rate: allarmi lanciati senza evento target successivo
        # Approssimazione: FAR = 1 - hit_rate (semplificazione conservativa)
        false_alarm_rate = max(0.0, 1.0 - hit_rate)
        
        # Frequenza attesa di allarmi per anno
        alerts_per_year = self.expected_frequency_alerts(m_threshold)
        
        # Penalità per eccesso di allarmi (vincolo operativo)
        alert_penalty = 0.0
        if alerts_per_year > self.max_alerts_per_year:
            excess_ratio = alerts_per_year / self.max_alerts_per_year
            alert_penalty = self.cost_base * (excess_ratio ** 2) * alerts_per_year
        
        # Net Value con loss non-lineare
        net_value = (
            self.benefit_hit * hit_rate -
            self.cost_false_alarm * (false_alarm_rate ** self.loss_exponent) -
            alert_penalty
        )
        
        return net_value
    
    def optimize_threshold(self, 
                          m_min: float = 1.0,
                          m_max: float = 6.0,
                          method: str = 'grid_search') -> Dict:
        """
        Ottimizza la soglia di magnitudo per massimizzare il Net Value.
        
        Parameters
        ----------
        m_min : float
            Magnitudo minima da considerare
        m_max : float
            Magnitudo massima da considerare
        method : str
            Metodo di ottimizzazione: 'grid_search' o 'scipy'
            
        Returns
        -------
        results : dict
            Dizionario con risultati dell'ottimizzazione
        """
        if self.lambda_rate is None:
            raise ValueError("Modello non fitted. Chiamare fit() prima.")
        
        if method == 'grid_search':
            # Grid search su un range di magnitudo
            magnitudes = np.linspace(m_min, m_max, 1000)
            net_values = [self.net_value(m) for m in magnitudes]
            
            best_idx = np.argmax(net_values)
            self.optimal_threshold = magnitudes[best_idx]
            self.optimal_net_value = net_values[best_idx]
            
            results = {
                'optimal_threshold': self.optimal_threshold,
                'optimal_net_value': self.optimal_net_value,
                'magnitudes': magnitudes,
                'net_values': np.array(net_values),
                'method': 'grid_search'
            }
            
        elif method == 'scipy':
            # Ottimizzazione con scipy
            def neg_net_value(m):
                return -self.net_value(m)
            
            result = minimize(
                neg_net_value,
                x0=[(m_min + m_max) / 2],
                bounds=[(m_min, m_max)],
                method='L-BFGS-B'
            )
            
            self.optimal_threshold = result.x[0]
            self.optimal_net_value = -result.fun
            
            results = {
                'optimal_threshold': self.optimal_threshold,
                'optimal_net_value': self.optimal_net_value,
                'success': result.success,
                'message': result.message,
                'method': 'scipy_optimize'
            }
        
        else:
            raise ValueError(f"Metodo {method} non supportato")
        
        self.optimization_results = results
        return results
    
    def generate_alerts(self, 
                       catalog: pd.DataFrame,
                       magnitude_col: str = 'magnitude',
                       time_col: str = 'time') -> pd.DataFrame:
        """
        Genera allarmi basati sulla soglia ottimale.
        
        Parameters
        ----------
        catalog : pd.DataFrame
            Catalogo sismico
        magnitude_col : str
            Colonna delle magnitudo
        time_col : str
            Colonna del tempo
            
        Returns
        -------
        alerts : pd.DataFrame
            DataFrame con gli allarmi generati
        """
        if self.optimal_threshold is None:
            raise ValueError("Soglia ottimale non calcolata. Chiamare optimize_threshold() prima.")
        
        catalog = catalog.copy()
        if not pd.api.types.is_datetime64_any_dtype(catalog[time_col]):
            catalog[time_col] = pd.to_datetime(catalog[time_col])
        
        # Identifica eventi sopra la soglia
        catalog['alert'] = catalog[magnitude_col] >= self.optimal_threshold
        catalog['alert_level'] = 'NONE'
        catalog.loc[catalog['alert'], 'alert_level'] = 'YELLOW'
        
        # Alert più severi per magnitudo molto alte
        high_threshold = self.optimal_threshold + 1.0
        catalog.loc[catalog[magnitude_col] >= high_threshold, 'alert_level'] = 'RED'
        
        alerts = catalog[catalog['alert']].copy()
        
        return alerts
    
    def validate_alerts(self, 
                       alerts: pd.DataFrame,
                       actual_events: pd.DataFrame,
                       tolerance_days: int = 7,
                       magnitude_col: str = 'magnitude',
                       time_col: str = 'time') -> Dict:
        """
        Valida le performance del sistema di allerta.
        
        Parameters
        ----------
        alerts : pd.DataFrame
            Allarmi generati dal modello
        actual_events : pd.DataFrame
            Eventi reali da prevedere (es. eventi significativi)
        tolerance_days : int
            Finestra di tolleranza temporale (giorni)
            
        Returns
        -------
        metrics : dict
            Metriche di validazione (hit rate, FAR, skill score, net value)
        """
        if not pd.api.types.is_datetime64_any_dtype(alerts[time_col]):
            alerts = alerts.copy()
            alerts[time_col] = pd.to_datetime(alerts[time_col])
        
        if not pd.api.types.is_datetime64_any_dtype(actual_events[time_col]):
            actual_events = actual_events.copy()
            actual_events[time_col] = pd.to_datetime(actual_events[time_col])
        
        # Conta gli eventi significativi (es. M >= 3.0 o soglia specifica)
        significant_events = actual_events[actual_events[magnitude_col] >= 3.0]
        n_total_events = len(significant_events)
        
        if n_total_events == 0:
            return {
                'hit_rate': 0.0,
                'false_alarm_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'net_value_realized': 0.0,
                'n_hits': 0,
                'n_false_alarms': len(alerts),
                'n_misses': 0,
                'n_correct_negatives': 0
            }
        
        # Conta hit e false alarms
        n_hits = 0
        n_false_alarms = 0
        
        for _, alert_row in alerts.iterrows():
            alert_time = alert_row[time_col]
            
            # Controlla se c'è un evento significativo entro la finestra
            time_window_start = alert_time - pd.Timedelta(days=tolerance_days)
            time_window_end = alert_time + pd.Timedelta(days=tolerance_days)
            
            events_in_window = significant_events[
                (significant_events[time_col] >= time_window_start) &
                (significant_events[time_col] <= time_window_end)
            ]
            
            if len(events_in_window) > 0:
                n_hits += 1
            else:
                n_false_alarms += 1
        
        n_misses = n_total_events - n_hits
        n_correct_negatives = max(0, len(alerts) - n_hits)
        
        # Calcola metriche
        hit_rate = n_hits / n_total_events if n_total_events > 0 else 0.0
        false_alarm_rate = n_false_alarms / len(alerts) if len(alerts) > 0 else 0.0
        precision = n_hits / (n_hits + n_false_alarms) if (n_hits + n_false_alarms) > 0 else 0.0
        recall = hit_rate
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Net Value realizzato
        net_value_realized = (
            self.benefit_hit * n_hits -
            self.cost_false_alarm * n_false_alarms -
            self.cost_base * len(alerts)
        )
        
        metrics = {
            'hit_rate': hit_rate,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'net_value_realized': net_value_realized,
            'net_value_per_alert': net_value_realized / len(alerts) if len(alerts) > 0 else 0.0,
            'n_hits': n_hits,
            'n_false_alarms': n_false_alarms,
            'n_misses': n_misses,
            'n_correct_negatives': n_correct_negatives,
            'n_total_significant_events': n_total_events,
            'n_total_alerts': len(alerts)
        }
        
        return metrics
    
    def sensitivity_analysis(self, 
                            benefit_range: Tuple[float, float] = (0.5, 2.0),
                            cost_range: Tuple[float, float] = (0.05, 0.2),
                            n_points: int = 10) -> pd.DataFrame:
        """
        Analisi di sensitività al variare di benefit e cost.
        
        Parameters
        ----------
        benefit_range : tuple
            Range di valori per il benefit (min, max)
        cost_range : tuple
            Range di valori per il cost (min, max)
        n_points : int
            Numero di punti per ogni dimensione
            
        Returns
        -------
        sensitivity_df : pd.DataFrame
            DataFrame con i risultati dell'analisi di sensitività
        """
        benefits = np.linspace(benefit_range[0], benefit_range[1], n_points)
        costs = np.linspace(cost_range[0], cost_range[1], n_points)
        
        results = []
        
        for b in benefits:
            for c in costs:
                # Aggiorna parametri
                self.benefit_hit = b
                self.cost_false_alarm = c
                
                # Ricalcola soglia ottimale
                opt_result = self.optimize_threshold(method='grid_search')
                
                results.append({
                    'benefit_hit': b,
                    'cost_false_alarm': c,
                    'ratio_b_c': b / c if c > 0 else np.inf,
                    'optimal_threshold': opt_result['optimal_threshold'],
                    'optimal_net_value': opt_result['optimal_net_value'],
                    'prob_exceedance_at_threshold': self.probability_exceedance(opt_result['optimal_threshold'])
                })
        
        # Ripristina parametri originali
        self.benefit_hit = benefit_range[0]
        self.cost_false_alarm = cost_range[0]
        
        sensitivity_df = pd.DataFrame(results)
        return sensitivity_df
    
    def get_summary(self) -> Dict:
        """
        Restituisce un riassunto del modello fitted.
        
        Returns
        -------
        summary : dict
            Dizionario con i parametri e risultati principali
        """
        summary = {
            'model_type': 'Actuarial Frequency-Magnitude',
            'fitted': self.lambda_rate is not None,
            'parameters': {
                'lambda_rate (daily)': self.lambda_rate,
                'b_value': self.b_value,
                'a_value': self.a_value,
                'mean_magnitude': self.mean_magnitude,
                'min_magnitude': self.min_magnitude,
                'time_window_days': self.time_window_days
            },
            'loss_function_params': {
                'benefit_hit': self.benefit_hit,
                'cost_false_alarm': self.cost_false_alarm,
                'cost_base': self.cost_base
            },
            'optimization': {
                'optimal_threshold': self.optimal_threshold,
                'optimal_net_value': self.optimal_net_value,
                'status': 'completed' if self.optimization_results is not None else 'not_run'
            }
        }
        
        return summary


def run_actuarial_model(catalog_path: str = '/workspace/data/processed/catalog_clean.csv',
                        output_dir: str = '/workspace/data/processed',
                        benefit_hit: float = 10.0,
                        cost_false_alarm: float = 1.0,
                        target_magnitude: float = 3.0,
                        max_alerts_per_year: int = 50,
                        save_results: bool = True) -> Dict:
    """
    Esegue l'intero pipeline del modello attuariale freq*magn con Net Value Loss.
    
    Parameters
    ----------
    catalog_path : str
        Percorso al catalogo sismico pulito
    output_dir : str
        Directory per salvare i risultati
    benefit_hit : float
        Beneficio per hit (valore tipico: 10-100)
    cost_false_alarm : float
        Costo per falso allarme (valore tipico: 1-5)
    target_magnitude : float
        Magnitudo target degli eventi da prevedere (default: 3.0)
    max_alerts_per_year : int
        Numero massimo di allarmi consentiti per anno
    save_results : bool
        Se salvare i risultati su file
        
    Returns
    -------
    results : dict
        Dizionario con tutti i risultati
    """
    print("=" * 70)
    print("ACTUARIAL FREQUENCY-MAGNITUDE MODEL WITH NET VALUE LOSS")
    print("=" * 70)
    
    # Carica il catalogo
    print(f"\n[1/6] Caricamento catalogo da {catalog_path}...")
    catalog = pd.read_csv(catalog_path)
    print(f"  - Eventi totali: {len(catalog)}")
    print(f"  - Periodo: {catalog['time'].min()} - {catalog['time'].max()}")
    print(f"  - Magnitudo range: [{catalog['magnitude'].min()}, {catalog['magnitude'].max()}]")
    
    # Conta eventi target
    n_target = len(catalog[catalog['magnitude'] >= target_magnitude])
    print(f"  - Eventi target (M >= {target_magnitude}): {n_target}")
    
    # Inizializza e fit del modello
    print(f"\n[2/6] Fit del modello attuariale...")
    model = ActuarialFrequencyMagnitudeModel(
        benefit_hit=benefit_hit,
        cost_false_alarm=cost_false_alarm,
        time_window_days=7,
        min_magnitude=1.0,
        target_magnitude=target_magnitude,
        max_alerts_per_year=max_alerts_per_year,
        loss_exponent=1.5
    )
    model.fit(catalog)
    
    summary = model.get_summary()
    print(f"  - Lambda rate (giornaliero): {summary['parameters']['lambda_rate (daily)']:.4f}")
    print(f"  - b-value: {summary['parameters']['b_value']:.3f}")
    print(f"  - a-value: {summary['parameters']['a_value']:.3f}")
    print(f"  - Magnitudo media: {summary['parameters']['mean_magnitude']:.3f}")
    print(f"  - Target magnitude: {target_magnitude}")
    print(f"  - Max alerts/year: {max_alerts_per_year}")
    
    # Ottimizzazione soglia
    print(f"\n[3/6] Ottimizzazione soglia per massimizzare Net Value...")
    opt_results = model.optimize_threshold(method='grid_search', m_min=1.0, m_max=5.0)
    print(f"  - Soglia ottimale: M = {opt_results['optimal_threshold']:.3f}")
    print(f"  - Net Value ottimale: {opt_results['optimal_net_value']:.4f}")
    print(f"  - Probabilità exceedance: {model.probability_exceedance(opt_results['optimal_threshold']):.4f}")
    print(f"  - Alert/anno attesi: {model.expected_frequency_alerts(opt_results['optimal_threshold']):.1f}")
    
    # Genera allarmi
    print(f"\n[4/6] Generazione allarmi con soglia ottimale...")
    alerts = model.generate_alerts(catalog)
    print(f"  - Totale allarmi generati: {len(alerts)}")
    print(f"  - Allarmi YELLOW (M >= {opt_results['optimal_threshold']:.2f}): {(alerts['alert_level'] == 'YELLOW').sum()}")
    print(f"  - Allarmi RED (M >= {opt_results['optimal_threshold'] + 1.0:.2f}): {(alerts['alert_level'] == 'RED').sum()}")
    
    # Validazione
    print(f"\n[5/6] Validazione performance...")
    validation_metrics = model.validate_alerts(alerts, catalog, tolerance_days=7)
    print(f"  - Hit Rate: {validation_metrics['hit_rate']*100:.1f}%")
    print(f"  - False Alarm Rate: {validation_metrics['false_alarm_rate']*100:.1f}%")
    print(f"  - Precision: {validation_metrics['precision']:.3f}")
    print(f"  - F1 Score: {validation_metrics['f1_score']:.3f}")
    print(f"  - Net Value Realizzato: {validation_metrics['net_value_realized']:.2f}")
    print(f"  - Hit: {validation_metrics['n_hits']}, False Alarms: {validation_metrics['n_false_alarms']}, Misses: {validation_metrics['n_misses']}")
    
    # Analisi di sensitività
    print(f"\n[6/6] Analisi di sensitività...")
    sensitivity_df = model.sensitivity_analysis(
        benefit_range=(5.0, 20.0),
        cost_range=(0.5, 3.0),
        n_points=6
    )
    print(f"  - Testate {len(sensitivity_df)} combinazioni di benefit/cost")
    print(f"  - Range soglie ottimali: [{sensitivity_df['optimal_threshold'].min():.3f}, {sensitivity_df['optimal_threshold'].max():.3f}]")
    print(f"  - Range Net Value: [{sensitivity_df['optimal_net_value'].min():.3f}, {sensitivity_df['optimal_net_value'].max():.3f}]")
    
    # Salva risultati
    if save_results:
        print(f"\n[SAVE] Salvataggio risultati in {output_dir}...")
        
        # Salva allarmi
        alerts_path = f"{output_dir}/actuarial_alerts.csv"
        alerts.to_csv(alerts_path, index=False)
        print(f"  - Allarmi: {alerts_path}")
        
        # Salva curva Net Value
        net_value_df = pd.DataFrame({
            'magnitude_threshold': opt_results['magnitudes'],
            'net_value': opt_results['net_values']
        })
        net_value_path = f"{output_dir}/actuarial_net_value_curve.csv"
        net_value_df.to_csv(net_value_path, index=False)
        print(f"  - Curva Net Value: {net_value_path}")
        
        # Salva analisi di sensitività
        sensitivity_path = f"{output_dir}/actuarial_sensitivity_analysis.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False)
        print(f"  - Sensitivity Analysis: {sensitivity_path}")
        
        # Salva metriche di validazione
        metrics_df = pd.DataFrame([validation_metrics])
        metrics_path = f"{output_dir}/actuarial_validation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  - Validation Metrics: {metrics_path}")
        
        # Salva summary
        import json
        summary_path = f"{output_dir}/actuarial_model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  - Model Summary: {summary_path}")
    
    # Compila risultati finali
    results = {
        'model': model,
        'summary': summary,
        'optimization_results': opt_results,
        'alerts': alerts,
        'validation_metrics': validation_metrics,
        'sensitivity_analysis': sensitivity_df,
        'files_saved': {
            'alerts': alerts_path if save_results else None,
            'net_value_curve': net_value_path if save_results else None,
            'sensitivity_analysis': sensitivity_path if save_results else None,
            'validation_metrics': metrics_path if save_results else None,
            'model_summary': summary_path if save_results else None
        }
    }
    
    print("\n" + "=" * 70)
    print("MODELLO ATTUARIALE COMPLETATO CON SUCCESSO")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    # Esegui il modello
    results = run_actuarial_model(
        catalog_path='/workspace/data/processed/catalog_clean.csv',
        output_dir='/workspace/data/processed',
        benefit_hit=10.0,
        cost_false_alarm=1.0,
        target_magnitude=3.0,
        max_alerts_per_year=50,
        save_results=True
    )
    
    # Stampa riepilogo finale
    print("\n\nRIEPILOGO FINALE:")
    print("-" * 70)
    print(f"Soglia ottimale di magnitudo: M = {results['optimization_results']['optimal_threshold']:.3f}")
    print(f"Net Value massimo teorico: {results['optimization_results']['optimal_net_value']:.4f}")
    print(f"\nPerformance su dati storici:")
    print(f"  Hit Rate: {results['validation_metrics']['hit_rate']*100:.1f}%")
    print(f"  False Alarm Rate: {results['validation_metrics']['false_alarm_rate']*100:.1f}%")
    print(f"  Precision: {results['validation_metrics']['precision']:.3f}")
    print(f"  Net Value Realizzato: {results['validation_metrics']['net_value_realized']:.2f}")
    print(f"\nInterpretazione:")
    print(f"  - Il modello prevede eventi con M >= {results['model'].target_magnitude}")
    print(f"  - Soglia di allerta ottimale: M >= {results['optimization_results']['optimal_threshold']:.2f}")
    print(f"  - Rapporto Benefit/Cost: {results['model'].benefit_hit / results['model'].cost_false_alarm:.1f}")
    print(f"  - Allarmi totali generati: {len(results['alerts'])} in {results['model'].time_span_years:.1f} anni")
    print(f"  - Frequenza allarmi: {len(results['alerts'])/results['model'].time_span_years:.1f} per anno")

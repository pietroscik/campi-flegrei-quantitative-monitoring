"""
Alert Engine for Campi Flegrei Monitoring System

This module evaluates analysis results and sends notifications
via Telegram or other channels when alert conditions are met.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Alert evaluation and notification engine.
    
    Evaluates early warning system results and sends notifications
    based on configurable thresholds and persistence rules.
    """
    
    def __init__(self, state_file: str = "data/processed/.last_alert_state.json"):
        """
        Initialize the alert engine.
        
        Parameters
        ----------
        state_file : str
            Path to JSON file storing last alert state for deduplication
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load previous alert state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load state file: {e}")
        
        return {
            "last_alert_time": None,
            "last_alert_level": None,
            "consecutive_alerts": 0,
            "notification_sent": False,
            "alert_history": []
        }
    
    def _save_state(self):
        """Save current alert state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.last_state, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save state file: {e}")
    
    def evaluate_alert_level(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate current alert level from early warning data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from early warning system with 'state' and 'alert_flag' columns
            
        Returns
        -------
        dict
            Alert evaluation results including level, confidence, and details
        """
        if df is None or len(df) == 0:
            return {
                "level": "UNKNOWN",
                "confidence": 0.0,
                "details": "No data available",
                "triggered": False
            }
        
        # Get latest record
        latest = df.iloc[-1]
        latest_time = latest.get('time', datetime.now())
        
        # Current state from classification
        current_state = latest.get("state", "NORMAL")
        alert_flag = latest.get("alert_flag", 0)
        
        # Persistence check - how many consecutive alerts
        recent_alerts = df.tail(7)["alert_flag"].sum()  # Last 7 records
        persistence = recent_alerts / 7.0
        
        # Trend analysis
        trend = latest.get("trend", 0.0)
        trend_direction = "increasing" if trend > 0 else "stable" if trend == 0 else "decreasing"
        
        # Determine alert level
        level_map = {
            "NORMAL": 0,
            "ELEVATED": 1,
            "HIGH": 2,
            "CRITICAL": 3
        }
        
        level_value = level_map.get(current_state, 0)
        
        # Build alert details
        details = {
            "timestamp": str(latest_time),
            "state": current_state,
            "alert_flag": bool(alert_flag),
            "persistence": round(persistence, 2),
            "trend": round(trend, 4),
            "trend_direction": trend_direction,
            "unrest_index": round(latest.get("unrest_index", 0), 4) if "unrest_index" in latest else None,
            "recent_alerts_count": int(recent_alerts)
        }
        
        # Determine if notification should be triggered
        # Only trigger on HIGH or CRITICAL with sufficient persistence
        triggered = (
            level_value >= 2 and  # HIGH or CRITICAL
            alert_flag == 1 and
            persistence >= 0.5  # At least 50% of recent records show alert
        )
        
        # Confidence score based on multiple factors
        confidence = 0.0
        if triggered:
            confidence = min(1.0, (
                0.3 * (level_value / 3.0) +  # State severity
                0.4 * persistence +           # Persistence
                0.3 * (1.0 if trend > 0 else 0.5)  # Trend confirmation
            ))
        
        return {
            "level": current_state,
            "level_value": level_value,
            "confidence": round(confidence, 2),
            "triggered": triggered,
            "details": details
        }
    
    def should_notify(self, alert_result: Dict[str, Any]) -> bool:
        """
        Determine if a notification should be sent (deduplication logic).
        
        Parameters
        ----------
        alert_result : dict
            Result from evaluate_alert_level()
            
        Returns
        -------
        bool
            True if notification should be sent
        """
        if not alert_result["triggered"]:
            return False
        
        current_level = alert_result["level"]
        last_level = self.last_state.get("last_alert_level")
        last_alert_time = self.last_state.get("last_alert_time")
        
        # Always notify on first alert or level escalation
        if last_level is None:
            return True
        
        if current_level in ["CRITICAL"] and last_level != "CRITICAL":
            return True
        
        if current_level == "HIGH" and last_level in ["NORMAL", "ELEVATED"]:
            return True
        
        # Time-based deduplication (avoid spam)
        if last_alert_time:
            try:
                last_dt = datetime.fromisoformat(last_alert_time)
                hours_since_last = (datetime.now() - last_dt).total_seconds() / 3600
                
                # Different cooldowns based on severity
                cooldown_hours = {
                    "CRITICAL": 1,   # Notify every hour for critical
                    "HIGH": 6,       # Notify every 6 hours for high
                    "ELEVATED": 24,  # Once per day for elevated
                    "NORMAL": 24     # Once per day for status updates
                }
                
                cooldown = cooldown_hours.get(current_level, 24)
                
                if hours_since_last < cooldown:
                    logger.info(f"Notification suppressed: only {hours_since_last:.1f}h since last alert (cooldown: {cooldown}h)")
                    return False
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing last alert time: {e}")
        
        return True
    
    def format_notification_message(self, alert_result: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Format a human-readable notification message.
        
        Parameters
        ----------
        alert_result : dict
            Result from evaluate_alert_level()
        config : dict
            Configuration dictionary
            
        Returns
        -------
        str
            Formatted notification message
        """
        details = alert_result["details"]
        level = alert_result["level"]
        
        # Emoji based on severity
        emoji_map = {
            "NORMAL": "✅",
            "ELEVATED": "⚠️",
            "HIGH": "🟠",
            "CRITICAL": "🚨"
        }
        emoji = emoji_map.get(level, "ℹ️")
        
        message = f"""
{emoji} *Campi Flegrei Monitoring Alert*

*Status:* {level}
*Time:* {details['timestamp']}
*Confidence:* {alert_result['confidence']*100:.0f}%

*Details:*
• Unrest Index: {details.get('unrest_index', 'N/A')}
• Trend: {details['trend_direction']} ({details['trend']:+.4f})
• Persistence: {details['persistence']*100:.0f}% (last 7 cycles)
• Recent Alerts: {details['recent_alerts_count']}/7

*Assessment:*
{self._get_assessment_text(level, details)}

---
Campi Flegrei Monitoring System
""".strip()
        
        return message
    
    def _get_assessment_text(self, level: str, details: Dict[str, Any]) -> str:
        """Get assessment text based on alert level."""
        assessments = {
            "NORMAL": "System parameters within normal ranges. No immediate concerns.",
            "ELEVATED": "Slight increase in unrest indicators. Continued monitoring recommended.",
            "HIGH": "Significant anomaly detected. Persistent unrest with positive trend. Close attention required.",
            "CRITICAL": "Critical alert triggered. Multiple indicators show sustained anomalous behavior. Immediate review recommended."
        }
        return assessments.get(level, "Unable to assess current situation.")
    
    def send_telegram_notification(self, message: str, config: Dict[str, Any]) -> bool:
        """
        Send notification via Telegram Bot API.
        
        Parameters
        ----------
        message : str
            Message to send
        config : dict
            Configuration with Telegram settings
            
        Returns
        -------
        bool
            True if notification sent successfully
        """
        import urllib.request
        import urllib.error
        
        token = os.getenv("TELEGRAM_BOT_TOKEN") or config.get("alerts", {}).get("telegram_bot_token")
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or config.get("alerts", {}).get("telegram_chat_id")
        
        if not token or not chat_id:
            logger.warning("Telegram credentials not configured. Skipping notification.")
            return False
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if result.get("ok"):
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {result}")
                    return False
                    
        except urllib.error.URLError as e:
            logger.error(f"Network error sending Telegram notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram notification: {e}")
            return False
    
    def check_and_notify(
        self,
        warning_df: pd.DataFrame,
        send_notification: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main method: evaluate alerts and send notifications if needed.
        
        Parameters
        ----------
        warning_df : pd.DataFrame
            Early warning system output DataFrame
        send_notification : bool
            Whether to actually send notifications
        config : dict, optional
            Configuration dictionary
            
        Returns
        -------
        dict
            Results including alert status and notification outcome
        """
        config = config or {}
        
        # Evaluate current alert level
        alert_result = self.evaluate_alert_level(warning_df)
        
        result = {
            "alert_triggered": alert_result["triggered"],
            "alert_details": alert_result["details"],
            "notification_sent": False,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        if not alert_result["triggered"]:
            # Update state even if no alert (for tracking)
            self.last_state["last_alert_time"] = datetime.now().isoformat()
            self.last_state["last_alert_level"] = alert_result["level"]
            self.last_state["consecutive_alerts"] = 0
            self._save_state()
            
            return result
        
        # Check if we should notify (deduplication)
        should_notify_flag = self.should_notify(alert_result)
        
        if should_notify_flag and send_notification:
            # Format and send notification
            message = self.format_notification_message(alert_result, config)
            
            notification_success = self.send_telegram_notification(message, config)
            result["notification_sent"] = notification_success
            
            if notification_success:
                self.last_state["consecutive_alerts"] += 1
            else:
                logger.warning("Notification failed")
        elif should_notify_flag and not send_notification:
            logger.info("Notification suppressed by --no-notifications flag")
            result["notification_sent"] = False
        else:
            logger.info("Notification suppressed by deduplication logic")
            result["notification_sent"] = False
        
        # Update state
        self.last_state["last_alert_time"] = datetime.now().isoformat()
        self.last_state["last_alert_level"] = alert_result["level"]
        self.last_state["notification_sent"] = result["notification_sent"]
        
        # Keep history (last 100 alerts)
        self.last_state["alert_history"].append({
            "timestamp": datetime.now().isoformat(),
            "level": alert_result["level"],
            "notified": result["notification_sent"]
        })
        self.last_state["alert_history"] = self.last_state["alert_history"][-100:]
        
        self._save_state()
        
        return result


# Legacy function for backward compatibility
def check_alerts(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Legacy function to check alerts (backward compatible).
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Alert check results
    """
    engine = AlertEngine()
    
    try:
        df = pd.read_csv("data/processed/early_warning_system.csv")
        result = engine.check_and_notify(df, send_notification=True)
        
        if result["alert_triggered"]:
            print("🚨 ALERT TRIGGERED")
            print(json.dumps(result["alert_details"], indent=2))
        else:
            print("✓ No alerts - system normal")
        
        return result
        
    except FileNotFoundError:
        print("Error: early_warning_system.csv not found. Run the analysis first.")
        return {"error": "Data file not found"}
    except Exception as e:
        print(f"Error checking alerts: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_alerts()

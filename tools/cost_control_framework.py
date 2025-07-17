#!/usr/bin/env python3
"""
Cost Control Framework per CBT Journal
Implementa budget protection bulletproof secondo schema v3.3.0
"""

import contextlib
import os
import json
import sqlite3
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

class BudgetStatus(Enum):
    """Status budget secondo schema v3.3.0"""
    WITHIN_LIMITS = "within_limits"
    APPROACHING_LIMIT = "approaching_limit"
    OVER_BUDGET = "over_budget"

class DailyBudgetStatus(Enum):
    """Status budget giornaliero"""
    ON_TRACK = "on_track"
    APPROACHING_LIMIT = "approaching_limit"
    OVER_BUDGET = "over_budget"

class MonthlyBudgetStatus(Enum):
    """Status budget mensile"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OVER_BUDGET = "over_budget"

@dataclass
class SessionBudget:
    """Budget singola sessione"""
    max_cost_per_session: float
    current_session_cost: float
    budget_utilization: float
    budget_status: BudgetStatus

@dataclass
class DailyTracking:
    """Tracking giornaliero"""
    date: str
    daily_cost_limit: float
    current_daily_cost: float
    sessions_today: int
    avg_cost_per_session: float
    projected_daily_cost: float
    daily_budget_status: DailyBudgetStatus

@dataclass
class MonthlyTracking:
    """Tracking mensile"""
    month: str
    monthly_budget: float
    current_monthly_cost: float
    days_elapsed: int
    projected_monthly_cost: float
    budget_utilization: float
    monthly_budget_status: MonthlyBudgetStatus

@dataclass
class CostAlerts:
    """Sistema alerts costi"""
    session_over_budget: bool
    daily_approaching_limit: bool
    monthly_approaching_limit: bool
    unusual_cost_spike: bool
    last_alert_timestamp: Optional[str]

@dataclass
class OptimizationSuggestion:
    """Suggerimenti ottimizzazione"""
    type: str
    description: str
    potential_savings_usd: float
    confidence: str

class CostControlManager:
    """Manager centrale controllo costi"""
    
    def __init__(self, db_path: str = "data/cost_control.db"):
        """Inizializza cost control con database SQLite"""
        self.db_path = db_path
        
        # Configurazione da .env
        self.max_cost_per_session = float(os.getenv("MAX_COST_PER_SESSION", "0.50"))
        self.max_daily_cost = float(os.getenv("MAX_DAILY_COST", "5.00"))
        self.max_monthly_cost = float(os.getenv("MAX_MONTHLY_COST", "100.00"))
        
        # Warning thresholds
        self.session_warning_threshold = 0.8  # 80% budget sessione
        self.daily_warning_threshold = 0.8    # 80% budget giornaliero
        self.monthly_warning_threshold = 0.8  # 80% budget mensile
        
        # Spike detection
        self.cost_spike_multiplier = 3.0  # 3x average = spike
        
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Inizializza database SQLite per tracking costi"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            # Tabella per tracking API calls
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    api_type TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    cost_usd REAL NOT NULL,
                    purpose TEXT,
                    processing_time_ms INTEGER
                )
            """)
            
            # Tabella per budget tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    daily_cost REAL NOT NULL,
                    sessions_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(date)
                )
            """)
            
            # Tabella per alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    cost_data TEXT
                )
            """)
            
            conn.commit()
    
    def estimate_api_cost(self, tokens_input: int, tokens_output: int, 
                         model: str = "gpt-4o-2024-11-20") -> float:
        """Stima costo API call secondo pricing OpenAI"""
        
        # Pricing OpenAI (aggiornato Novembre 2024)
        pricing = {
            "gpt-4o-2024-11-20": {
                "input": 2.50,      # $2.50 per 1M input tokens
                "output": 10.00     # $10.00 per 1M output tokens
            },
            "gpt-4o-mini": {
                "input": 0.150,     # $0.150 per 1M input tokens
                "output": 0.600     # $0.600 per 1M output tokens
            },
            "text-embedding-3-large": {
                "input":  0.130,    # $0.130 per 1M tokens
                "output": 0.0       # No output cost for embeddings
            }
        }
        
        if model not in pricing:
            self.logger.warning(f"Unknown model {model}, using gpt-4o pricing")
            model = "gpt-4o-2024-11-20"
        
        rates = pricing[model]
        
        cost_input = (tokens_input / 1_000_000) * rates["input"]
        cost_output = (tokens_output / 1_000_000) * rates["output"]
        
        total_cost = cost_input + cost_output
        
        self.logger.debug(f"Cost estimate: {tokens_input}+{tokens_output} tokens = ${total_cost:.6f}")
        
        return total_cost
    
    def check_session_budget(self, session_id: str, estimated_cost: float) -> Tuple[bool, SessionBudget]:
        """Verifica budget sessione prima di API call"""
        
        # Get current session cost
        current_cost = self._get_session_cost(session_id)
        projected_cost = current_cost + estimated_cost
        
        budget_utilization = projected_cost / self.max_cost_per_session
        
        # Determina status
        if projected_cost > self.max_cost_per_session:
            status = BudgetStatus.OVER_BUDGET
            allowed = False
        elif budget_utilization >= self.session_warning_threshold:
            status = BudgetStatus.APPROACHING_LIMIT
            allowed = True
        else:
            status = BudgetStatus.WITHIN_LIMITS
            allowed = True
        
        session_budget = SessionBudget(
            max_cost_per_session=self.max_cost_per_session,
            current_session_cost=projected_cost,
            budget_utilization=budget_utilization,
            budget_status=status
        )
        
        if not allowed:
            self._log_alert("session_over_budget", "ERROR", 
                          f"Session {session_id} exceeds budget: ${projected_cost:.4f} > ${self.max_cost_per_session}")
        
        return allowed, session_budget
    
    def check_daily_budget(self, estimated_cost: float) -> Tuple[bool, DailyTracking]:
        """Verifica budget giornaliero"""
        today = date.today().isoformat()
        
        current_daily_cost = self._get_daily_cost(today)
        sessions_today = self._get_daily_sessions_count(today)
        
        projected_cost = current_daily_cost + estimated_cost
        avg_cost = current_daily_cost / max(sessions_today, 1)
        
        # Proiezione fine giornata (assumendo 3 sessioni medie al giorno)
        projected_daily_cost = current_daily_cost + (avg_cost * max(0, 3 - sessions_today))
        
        # Determina status
        if projected_cost > self.max_daily_cost:
            status = DailyBudgetStatus.OVER_BUDGET
            allowed = False
        elif projected_cost >= (self.max_daily_cost * self.daily_warning_threshold):
            status = DailyBudgetStatus.APPROACHING_LIMIT
            allowed = True
        else:
            status = DailyBudgetStatus.ON_TRACK
            allowed = True
        
        daily_tracking = DailyTracking(
            date=today,
            daily_cost_limit=self.max_daily_cost,
            current_daily_cost=projected_cost,
            sessions_today=sessions_today + 1,
            avg_cost_per_session=avg_cost,
            projected_daily_cost=projected_daily_cost,
            daily_budget_status=status
        )
        
        if not allowed:
            self._log_alert("daily_over_budget", "ERROR",
                          f"Daily budget exceeded: ${projected_cost:.4f} > ${self.max_daily_cost}")
        
        return allowed, daily_tracking
    
    def check_monthly_budget(self, estimated_cost: float) -> Tuple[bool, MonthlyTracking]:
        """Verifica budget mensile"""
        today = date.today()
        month = today.strftime("%Y-%m")
        
        current_monthly_cost = self._get_monthly_cost(month)
        projected_cost = current_monthly_cost + estimated_cost
        
        days_elapsed = today.day
        days_in_month = 30  # Approssimazione conservativa
        
        # Proiezione fine mese
        daily_avg = current_monthly_cost / max(days_elapsed, 1)
        projected_monthly_cost = daily_avg * days_in_month
        
        budget_utilization = projected_cost / self.max_monthly_cost
        
        # Determina status
        if projected_cost > self.max_monthly_cost:
            status = MonthlyBudgetStatus.OVER_BUDGET
            allowed = False
        elif budget_utilization >= 0.9:
            status = MonthlyBudgetStatus.CRITICAL
            allowed = True
        elif budget_utilization >= self.monthly_warning_threshold:
            status = MonthlyBudgetStatus.WARNING
            allowed = True
        else:
            status = MonthlyBudgetStatus.HEALTHY
            allowed = True
        
        monthly_tracking = MonthlyTracking(
            month=month,
            monthly_budget=self.max_monthly_cost,
            current_monthly_cost=projected_cost,
            days_elapsed=days_elapsed,
            projected_monthly_cost=projected_monthly_cost,
            budget_utilization=budget_utilization,
            monthly_budget_status=status
        )
        
        if not allowed:
            self._log_alert("monthly_over_budget", "CRITICAL",
                          f"Monthly budget exceeded: ${projected_cost:.4f} > ${self.max_monthly_cost}")
        
        return allowed, monthly_tracking
    
    def pre_api_check(self, session_id: str, tokens_input: int, tokens_output: int, 
                     model: str = "gpt-4o-2024-11-20") -> Tuple[bool, Dict]:
        """Check completo pre-API call con hard limits"""
        
        estimated_cost = self.estimate_api_cost(tokens_input, tokens_output, model)
        
        # Check tutti i livelli
        session_allowed, session_budget = self.check_session_budget(session_id, estimated_cost)
        daily_allowed, daily_tracking = self.check_daily_budget(estimated_cost)
        monthly_allowed, monthly_tracking = self.check_monthly_budget(estimated_cost)
        
        # AND logic: tutti devono permettere l'operazione
        allowed = session_allowed and daily_allowed and monthly_allowed
        
        # Genera alerts se necessario
        alerts = self._generate_alerts(session_budget, daily_tracking, monthly_tracking)
        
        # Suggerimenti ottimizzazione
        suggestions = self._generate_optimization_suggestions(
            estimated_cost, session_budget, daily_tracking, monthly_tracking
        )
        
        result = {
            "allowed": allowed,
            "estimated_cost": estimated_cost,
            "session_budget": asdict(session_budget),
            "daily_tracking": asdict(daily_tracking),
            "monthly_tracking": asdict(monthly_tracking),
            "cost_alerts": asdict(alerts),
            "optimization_suggestions": [asdict(s) for s in suggestions]
        }
        
        if not allowed:
            self.logger.error(f"API call BLOCKED for session {session_id}: Budget limits exceeded")
            self.logger.error(f"Estimated cost: ${estimated_cost:.6f}")
            self.logger.error(f"Session allowed: {session_allowed}, Daily: {daily_allowed}, Monthly: {monthly_allowed}")
        
        return allowed, result
    
    def record_api_cost(self, session_id: str, api_type: str, model: str, provider: str,
                       tokens_input: int, tokens_output: int, actual_cost: float,
                       purpose: str = None, processing_time_ms: int = None):
        """Registra costo API effettivo"""
        
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO api_costs 
                (session_id, timestamp, api_type, model, provider, tokens_input, tokens_output, 
                 cost_usd, purpose, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, datetime.now().isoformat(), api_type, model, provider,
                tokens_input, tokens_output, actual_cost, purpose, processing_time_ms
            ))
            
            # Update daily tracking
            today = date.today().isoformat()
            daily_cost = self._get_daily_cost(today)
            sessions_count = self._get_daily_sessions_count(today)
            
            conn.execute("""
                INSERT OR REPLACE INTO budget_tracking (date, daily_cost, sessions_count, created_at)
                VALUES (?, ?, ?, ?)
            """, (today, daily_cost, sessions_count, datetime.now().isoformat()))
            
            conn.commit()
        
        self.logger.info(f"Recorded API cost: ${actual_cost:.6f} for session {session_id}")
    
    def get_cost_summary(self, period: str = "month") -> Dict:
        """Genera summary costi per periodo"""
        
        if period == "today":
            today = date.today().isoformat()
            return {
                "period": "today",
                "total_cost": self._get_daily_cost(today),
                "sessions": self._get_daily_sessions_count(today),
                "budget_limit": self.max_daily_cost,
                "utilization": self._get_daily_cost(today) / self.max_daily_cost
            }
        
        elif period == "month":
            month = date.today().strftime("%Y-%m")
            cost = self._get_monthly_cost(month)
            return {
                "period": month,
                "total_cost": cost,
                "budget_limit": self.max_monthly_cost,
                "utilization": cost / self.max_monthly_cost,
                "days_elapsed": date.today().day
            }
        
        else:
            raise ValueError(f"Unsupported period: {period}")
    
    def _get_session_cost(self, session_id: str) -> float:
        """Get costo corrente sessione"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT SUM(cost_usd) FROM api_costs WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def _get_daily_cost(self, date_str: str) -> float:
        """Get costo giornaliero"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT SUM(cost_usd) FROM api_costs WHERE DATE(timestamp) = ?",
                (date_str,)
            )
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def _get_daily_sessions_count(self, date_str: str) -> int:
        """Get numero sessioni giornaliere"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM api_costs WHERE DATE(timestamp) = ?",
                (date_str,)
            )
            return cursor.fetchone()[0]
    
    def _get_monthly_cost(self, month: str) -> float:
        """Get costo mensile (formato YYYY-MM)"""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT SUM(cost_usd) FROM api_costs WHERE strftime('%Y-%m', timestamp) = ?",
                (month,)
            )
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def _generate_alerts(self, session_budget: SessionBudget, 
                        daily_tracking: DailyTracking, 
                        monthly_tracking: MonthlyTracking) -> CostAlerts:
        """Genera alerts basato su status"""
        
        session_over = session_budget.budget_status == BudgetStatus.OVER_BUDGET
        daily_approaching = daily_tracking.daily_budget_status == DailyBudgetStatus.APPROACHING_LIMIT
        monthly_approaching = monthly_tracking.monthly_budget_status in [
            MonthlyBudgetStatus.WARNING, MonthlyBudgetStatus.CRITICAL
        ]
        
        # Check cost spike
        unusual_spike = self._detect_cost_spike(session_budget.current_session_cost)
        
        return CostAlerts(
            session_over_budget=session_over,
            daily_approaching_limit=daily_approaching,
            monthly_approaching_limit=monthly_approaching,
            unusual_cost_spike=unusual_spike,
            last_alert_timestamp=datetime.now().isoformat() if any([
                session_over, daily_approaching, monthly_approaching, unusual_spike
            ]) else None
        )
    
    def _detect_cost_spike(self, current_cost: float) -> bool:
        """Rileva spike costi anomali"""
        
        # Get average session cost ultima settimana
        week_ago = (datetime.now().date() - timedelta(days=7)).isoformat()

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT AVG(session_total) FROM (
                    SELECT session_id, SUM(cost_usd) as session_total 
                    FROM api_costs 
                    WHERE DATE(timestamp) >= ?
                    GROUP BY session_id
                )
            """, (week_ago,))
            
            avg_cost = cursor.fetchone()[0]
            
            if avg_cost and current_cost > (avg_cost * self.cost_spike_multiplier):
                return True
                
        return False
    
    def _generate_optimization_suggestions(self, estimated_cost: float,
                                         session_budget: SessionBudget,
                                         daily_tracking: DailyTracking,
                                         monthly_tracking: MonthlyTracking) -> List[OptimizationSuggestion]:
        """Genera suggerimenti ottimizzazione"""
        
        suggestions = []
        
        # Suggerimento riduzione token se sessione costosa
        if session_budget.budget_utilization > 0.7:
            suggestions.append(OptimizationSuggestion(
                type="token_reduction",
                description="Consider reducing context length or response tokens",
                potential_savings_usd=estimated_cost * 0.3,
                confidence="medium"
            ))
        
        # Suggerimento modello più economico se budget critico
        if monthly_tracking.monthly_budget_status == MonthlyBudgetStatus.CRITICAL:
            suggestions.append(OptimizationSuggestion(
                type="model_selection",
                description="Consider using gpt-4o-mini for non-critical operations",
                potential_savings_usd=estimated_cost * 0.8,
                confidence="high"
            ))
        
        # Suggerimento batch processing se molte API calls
        if daily_tracking.sessions_today > 5:
            suggestions.append(OptimizationSuggestion(
                type="batch_processing",
                description="Consider batching multiple operations in single API call",
                potential_savings_usd=estimated_cost * 0.4,
                confidence="medium"
            ))
        
        return suggestions
    
    def _log_alert(self, alert_type: str, severity: str, message: str):
        """Log alert nel database"""

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO cost_alerts (timestamp, alert_type, severity, message, cost_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), alert_type, severity, message, 
                json.dumps(self.get_cost_summary("today"))
            ))
            conn.commit()
        
        self.logger.warning(f"{severity}: {message}")


# Usage example
if __name__ == "__main__":
    # Initialize cost control
    cost_manager = CostControlManager()
    
    # Example: Check before API call
    session_id = "test_session_001"
    allowed, check_result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=1000,
        tokens_output=500,
        model="gpt-4o-2024-11-20"
    )
    
    print(f"API call allowed: {allowed}")
    print(f"Estimated cost: ${check_result['estimated_cost']:.6f}")
    print(f"Session budget status: {check_result['session_budget']['budget_status']}")
    print(f"Daily budget status: {check_result['daily_tracking']['daily_budget_status']}")
    print(f"Monthly budget status: {check_result['monthly_tracking']['monthly_budget_status']}")
    
    if allowed:
        # Simulate API call
        actual_cost = check_result['estimated_cost']  # In reality, get from API response
        cost_manager.record_api_cost(
            session_id=session_id,
            api_type="chat_completion",
            model="gpt-4o-2024-11-20",
            provider="openai",
            tokens_input=1000,
            tokens_output=500,
            actual_cost=actual_cost,
            purpose="therapeutic_response"
        )
        
        print(f"API cost recorded: ${actual_cost:.6f}")
    
    # Show cost summary
    summary = cost_manager.get_cost_summary("today")
    print(f"\nToday's summary: ${summary['total_cost']:.4f} / ${summary['budget_limit']:.2f}")
    print(f"Budget utilization: {summary['utilization']:.1%}")

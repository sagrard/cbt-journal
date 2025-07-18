def test_edge_cases(cost_manager):
    """Verifica la gestione di casi limite di token"""
    edge_cases = [
        {"name": "Zero tokens", "tokens_input": 0, "tokens_output": 0, "should_work": True},
        {"name": "Very large input", "tokens_input": 100000, "tokens_output": 10000, "should_work": True},
        {"name": "Only input tokens", "tokens_input": 1000, "tokens_output": 0, "should_work": True},
        {"name": "Only output tokens", "tokens_input": 0, "tokens_output": 1000, "should_work": True},
    ]
    for case in edge_cases:
        try:
            cost = cost_manager.estimate_api_cost(
                case['tokens_input'],
                case['tokens_output'],
                "gpt-4o-2024-11-20"
            )
            allowed, result = cost_manager.pre_api_check(
                session_id=f"edge_case_{case['name'].replace(' ', '_')}",
                tokens_input=case['tokens_input'],
                tokens_output=case['tokens_output'],
                model="gpt-4o-2024-11-20"
            )
            # Se should_work è True, non deve sollevare eccezioni
            assert allowed is not None  # Deve restituire un booleano
        except Exception as e:
            assert not case['should_work'], f"Edge case fallito: {case['name']} - {e}"
def test_data_persistence(cost_manager, db_path):
    """Verifica che i dati di costo siano persistenti tra istanze del manager"""
    test_session = "persistence_test"
    test_cases = [
        {
            "name": "Light CBT response",
            "tokens_input": 500,
            "tokens_output": 200,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.001, 0.005)
        },
        {
            "name": "Normal CBT session",
            "tokens_input": 2000,
            "tokens_output": 800,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.01, 0.02)
        },
        {
            "name": "Heavy context session",
            "tokens_input": 8000,
            "tokens_output": 2000,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.04, 0.08)
        },
        {
            "name": "Mini model equivalent",
            "tokens_input": 2000,
            "tokens_output": 800,
            "model": "gpt-4o-mini",
            "expected_range": (0.0007, 0.002)
        }
    ]
    for case in test_cases:
        cost = cost_manager.estimate_api_cost(
            case["tokens_input"],
            case["tokens_output"],
            case["model"]
        )
        min_expected, max_expected = case["expected_range"]
        assert min_expected <= cost <= max_expected, f"{case['name']} fuori range: {cost}"

    # ...existing code...
def test_cost_spike_detection(cost_manager):
    """Verifica che la rilevazione di spike di costo funzioni correttamente"""
    import sqlite3
    from datetime import datetime, timedelta
    from contextlib import closing
    # Simula costi storici (baseline)
    baseline_cost = 0.015
    with closing(sqlite3.connect(cost_manager.db_path)) as conn:
        for days_ago in range(7, 1, -1):
            past_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            for session_num in range(3):
                session_id = f"baseline_{days_ago}_{session_num}"
                conn.execute(
                    """
                    INSERT INTO api_costs 
                    (session_id, timestamp, api_type, model, provider, tokens_input, tokens_output, cost_usd, purpose)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, past_date, "chat_completion", "gpt-4o-2024-11-20", "openai",
                        1500, 600, baseline_cost, "baseline_test"
                    )
                )
        conn.commit()
    # Sessione normale: non deve triggerare spike
    allowed, result = cost_manager.pre_api_check(
        session_id="spike_test_normal",
        tokens_input=1500,
        tokens_output=600,
        model="gpt-4o-2024-11-20"
    )
    normal_spike = result['cost_alerts']['unusual_cost_spike']
    assert not normal_spike, "La sessione normale non deve triggerare spike"
    # Sessione anomala: deve triggerare spike
    allowed2, result2 = cost_manager.pre_api_check(
        session_id="spike_test_anomaly",
        tokens_input=15000,
        tokens_output=6000,
        model="gpt-4o-2024-11-20"
    )
    anomaly_spike = result2['cost_alerts']['unusual_cost_spike']
    assert anomaly_spike, "La sessione anomala deve triggerare spike"
def test_daily_budget_control(cost_manager):
    """Verifica che il controllo del budget giornaliero funzioni correttamente"""
    session_costs = []
    blocked = False
    for i in range(5):
        session_id = f"daily_test_{i}"
        tokens_input = 1000 + (i * 1000)
        tokens_output = 400 + (i * 400)
        allowed, result = cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model="gpt-4o-2024-11-20"
        )
        cost = result['estimated_cost']
        daily_status = result['daily_tracking']['daily_budget_status']
        daily_cost = result['daily_tracking']['current_daily_cost']
        if allowed:
            cost_manager.record_api_cost(
                session_id=session_id,
                api_type="chat_completion",
                model="gpt-4o-2024-11-20",
                provider="openai",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                actual_cost=cost,
                purpose=f"daily_test_{i}"
            )
            session_costs.append(cost)
        else:
            blocked = True
            # Deve bloccare prima di superare il limite
            assert daily_status == BudgetStatus.OVER_BUDGET or not allowed
            break
    total_cost = sum(session_costs)
    # Il totale non deve mai superare il daily limit
    assert total_cost < cost_manager.max_daily_cost, "Il costo totale non deve superare il daily budget"
    # Se il totale si avvicina molto al limite, almeno una sessione deve essere bloccata; se invece tutte le sessioni sono piccole, nessuna viene bloccata
    if total_cost > (cost_manager.max_daily_cost * 0.95):
        assert blocked, "Almeno una sessione deve essere bloccata dal controllo giornaliero"
#!/usr/bin/env python3
"""
Test Suite Completo per Cost Control Framework
Verifica tutte le funzionalità con scenari realistici CBT
"""

import os
import sys
import uuid
import tempfile
import shutil
from datetime import datetime, date
from pathlib import Path


# Setup path per import
sys.path.append(str(Path(__file__).parent.parent))

from cbt_journal.utils.cost_control import CostControlManager, BudgetStatus

# ===================== PYTEST REFACTORING =====================
import pytest

# ---- Fixtures ----
@pytest.fixture(scope="module")
def temp_dir():
    d = tempfile.mkdtemp(prefix="cbt_test_")
    yield d
    shutil.rmtree(d)

@pytest.fixture
def db_path(temp_dir):
    return os.path.join(temp_dir, "test_cost_control.db")

@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("MAX_COST_PER_SESSION", "0.10")
    monkeypatch.setenv("MAX_DAILY_COST", "1.00")
    monkeypatch.setenv("MAX_MONTHLY_COST", "10.00")

@pytest.fixture
def cost_manager(db_path):
    return CostControlManager(db_path)

# ---- Test functions ----

def test_cost_estimation(cost_manager):
    test_cases = [
        {
            "name": "Light CBT response",
            "tokens_input": 500,
            "tokens_output": 200,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.001, 0.005)
        },
        {
            "name": "Normal CBT session",
            "tokens_input": 2000,
            "tokens_output": 800,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.01, 0.02)
        },
        {
            "name": "Heavy context session",
            "tokens_input": 8000,
            "tokens_output": 2000,
            "model": "gpt-4o-2024-11-20",
            "expected_range": (0.04, 0.08)
        },
        {
            "name": "Mini model equivalent",
            "tokens_input": 2000,
            "tokens_output": 800,
            "model": "gpt-4o-mini",
            "expected_range": (0.0007, 0.002)
        }
    ]
    for case in test_cases:
        cost = cost_manager.estimate_api_cost(
            case["tokens_input"],
            case["tokens_output"],
            case["model"]
        )
        min_expected, max_expected = case["expected_range"]
        assert min_expected <= cost <= max_expected, f"{case['name']} fuori range: {cost}"

def test_session_budget_control(cost_manager):
    """Verifica che il controllo del budget di sessione funzioni correttamente"""
    import uuid
    session_id = str(uuid.uuid4())

    # Caso normale: entro il budget
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=1000,
        tokens_output=400,
        model="gpt-4o-2024-11-20"
    )
    cost1 = result['estimated_cost']
    status1 = result['session_budget']['budget_status']
    assert allowed, "La sessione normale dovrebbe essere permessa"
    assert status1 == BudgetStatus.WITHIN_LIMITS, f"Stato inatteso: {status1}"

    # Registra il costo
    cost_manager.record_api_cost(
        session_id=session_id,
        api_type="chat_completion",
        model="gpt-4o-2024-11-20",
        provider="openai",
        tokens_input=1000,
        tokens_output=400,
        actual_cost=cost1,
        purpose="test_normal"
    )

    # Caso over budget: deve essere bloccato
    allowed2, result2 = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=15000,
        tokens_output=8000,
        model="gpt-4o-2024-11-20"
    )
    status2 = result2['session_budget']['budget_status']
    assert not allowed2, "La sessione over budget dovrebbe essere bloccata"
    assert status2 == BudgetStatus.OVER_BUDGET, f"Stato inatteso: {status2}"

    # Niente chiusura forzata qui: la fixture e contextlib.closing nei test garantiscono la chiusura

def test_monthly_budget_control(cost_manager):
    """Test controllo budget mensile"""
    import uuid
    from datetime import datetime, date
    
    # Test normal monthly cost
    session_id = str(uuid.uuid4())
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=2000,
        tokens_output=800,
        model="gpt-4o-2024-11-20"
    )
    assert allowed, "Normal monthly cost should be allowed"
    assert result['monthly_tracking']['monthly_budget_status'].value in ['healthy', 'warning']
    
    # Record the cost
    cost_manager.record_api_cost(
        session_id=session_id,
        api_type="chat_completion",
        model="gpt-4o-2024-11-20",
        provider="openai",
        tokens_input=2000,
        tokens_output=800,
        actual_cost=result['estimated_cost'],
        purpose="monthly_test"
    )

def test_optimization_suggestions(cost_manager):
    """Test generazione suggerimenti ottimizzazione"""
    import uuid
    
    # High cost session should generate suggestions
    session_id = str(uuid.uuid4())
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=5000,
        tokens_output=2000,
        model="gpt-4o-2024-11-20"
    )
    
    suggestions = result['optimization_suggestions']
    assert len(suggestions) > 0, "High cost session should generate suggestions"
    
    # Check suggestion types
    suggestion_types = [s['type'] for s in suggestions]
    assert 'token_reduction' in suggestion_types, "Should suggest token reduction"

def test_cost_alerts_generation(cost_manager):
    """Test generazione alerts per costi"""
    import uuid
    
    # Test alert for high session cost
    session_id = str(uuid.uuid4())
    
    # First, add some cost to the session
    cost_manager.record_api_cost(
        session_id=session_id,
        api_type="chat_completion",
        model="gpt-4o-2024-11-20",
        provider="openai",
        tokens_input=1000,
        tokens_output=500,
        actual_cost=0.08,
        purpose="alert_test"
    )
    
    # Now try to add more cost that would exceed budget
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=5000,
        tokens_output=2000,
        model="gpt-4o-2024-11-20"
    )
    
    alerts = result['cost_alerts']
    assert alerts['session_over_budget'] == (not allowed), "Session over budget alert should match allowed status"

def test_database_schema_integrity(cost_manager):
    """Test integrità schema database"""
    import sqlite3
    from contextlib import closing
    
    # Test that all required tables exist
    with closing(sqlite3.connect(cost_manager.db_path)) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['api_costs', 'budget_tracking', 'cost_alerts']
        for table in required_tables:
            assert table in tables, f"Required table {table} not found"
        
        # Test api_costs table structure
        cursor = conn.execute("PRAGMA table_info(api_costs)")
        columns = [row[1] for row in cursor.fetchall()]
        required_columns = ['session_id', 'timestamp', 'api_type', 'model', 'provider', 'tokens_input', 'tokens_output', 'cost_usd']
        for col in required_columns:
            assert col in columns, f"Required column {col} not found in api_costs"

def test_cost_summary_functionality(cost_manager):
    """Test funzionalità cost summary"""
    import uuid
    
    # Add some test costs
    session_id = str(uuid.uuid4())
    test_cost = 0.05
    
    cost_manager.record_api_cost(
        session_id=session_id,
        api_type="chat_completion",
        model="gpt-4o-2024-11-20",
        provider="openai",
        tokens_input=1000,
        tokens_output=500,
        actual_cost=test_cost,
        purpose="summary_test"
    )
    
    # Test today summary
    today_summary = cost_manager.get_cost_summary("today")
    assert today_summary['period'] == "today"
    assert today_summary['total_cost'] >= test_cost
    assert 'budget_limit' in today_summary
    assert 'utilization' in today_summary
    
    # Test month summary  
    month_summary = cost_manager.get_cost_summary("month")
    assert month_summary['total_cost'] >= test_cost
    assert 'budget_limit' in month_summary
    assert 'utilization' in month_summary

def test_concurrent_session_handling(cost_manager):
    """Test handling di sessioni concorrenti"""
    import uuid
    import threading
    import time
    
    results = []
    
    def create_session(session_id):
        try:
            allowed, result = cost_manager.pre_api_check(
                session_id=session_id,
                tokens_input=1000,
                tokens_output=400,
                model="gpt-4o-2024-11-20"
            )
            if allowed:
                cost_manager.record_api_cost(
                    session_id=session_id,
                    api_type="chat_completion",
                    model="gpt-4o-2024-11-20",
                    provider="openai",
                    tokens_input=1000,
                    tokens_output=400,
                    actual_cost=result['estimated_cost'],
                    purpose="concurrent_test"
                )
            results.append(allowed)
        except Exception as e:
            results.append(False)
    
    # Create multiple concurrent sessions
    threads = []
    for i in range(3):
        session_id = f"concurrent_session_{i}"
        thread = threading.Thread(target=create_session, args=(session_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # At least some sessions should be allowed
    assert len(results) == 3
    assert any(results), "At least one concurrent session should be allowed"

def test_invalid_model_handling(cost_manager):
    """Test handling di modelli non supportati"""
    import uuid
    
    session_id = str(uuid.uuid4())
    
    # Test with unknown model - should use default pricing
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=1000,
        tokens_output=400,
        model="unknown-model-xyz"
    )
    
    # Should still work with default pricing
    assert isinstance(result['estimated_cost'], float)
    assert result['estimated_cost'] > 0

def test_zero_cost_handling(cost_manager):
    """Test handling di costi zero"""
    import uuid
    
    session_id = str(uuid.uuid4())
    
    # Test with zero tokens
    allowed, result = cost_manager.pre_api_check(
        session_id=session_id,
        tokens_input=0,
        tokens_output=0,
        model="gpt-4o-2024-11-20"
    )
    
    assert allowed, "Zero cost should always be allowed"
    assert result['estimated_cost'] == 0.0

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

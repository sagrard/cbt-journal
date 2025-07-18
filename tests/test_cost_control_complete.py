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

class CostControlTestSuite:
    """Test suite completo per Cost Control"""
    
    def __init__(self):
        # Usa database temporaneo per test
        self.temp_dir = tempfile.mkdtemp(prefix="cbt_test_")
        self.db_path = os.path.join(self.temp_dir, "test_cost_control.db")
        
        # Inizializza con limiti test più bassi
        os.environ["MAX_COST_PER_SESSION"] = "0.10"  # $0.10 per session
        os.environ["MAX_DAILY_COST"] = "1.00"        # $1.00 per day
        os.environ["MAX_MONTHLY_COST"] = "10.00"     # $10.00 per month

        self.cost_manager = CostControlManager(self.db_path)
        
        print(f"🧪 Test environment initialized")
        print(f"📁 Database: {self.db_path}")
        print(f"💰 Session limit: ${self.cost_manager.max_cost_per_session}")
        print(f"📅 Daily limit: ${self.cost_manager.max_daily_cost}")
        print(f"📆 Monthly limit: ${self.cost_manager.max_monthly_cost}")
    
    def cleanup(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"🧹 Test environment cleaned up")
    
    def test_cost_estimation(self) -> bool:
        """Test 1: Verifica accuratezza stime costi"""
        print("\n" + "="*60)
        print("🧮 TEST 1: COST ESTIMATION ACCURACY")
        print("="*60)
        
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
        
        all_passed = True
        
        for case in test_cases:
            cost = self.cost_manager.estimate_api_cost(
                case["tokens_input"], 
                case["tokens_output"], 
                case["model"]
            )
            
            min_expected, max_expected = case["expected_range"]
            passed = min_expected <= cost <= max_expected
            
            status = "✅" if passed else "❌"
            print(f"{status} {case['name']}: ${cost:.6f} (expected ${min_expected:.3f}-${max_expected:.3f})")
            
            if not passed:
                all_passed = False
        
        print(f"\n📊 Cost estimation test: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    def test_session_budget_control(self) -> bool:
        """Test 2: Controllo budget sessione"""
        print("\n" + "="*60)
        print("💰 TEST 2: SESSION BUDGET CONTROL")
        print("="*60)
        
        session_id = str(uuid.uuid4())
        
        # Test caso normale (dentro budget)
        allowed, result = self.cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=1000,
            tokens_output=400,
            model="gpt-4o-2024-11-20"
        )
        
        cost1 = result['estimated_cost']
        status1 = result['session_budget']['budget_status']
        
        print(f"✅ Normal usage: ${cost1:.6f} - Status: {status1} - Allowed: {allowed}")
        
        if allowed and status1 == BudgetStatus.WITHIN_LIMITS:
            print("✅ Normal usage correctly allowed")
        else:
            print("❌ Normal usage should be allowed") 
            return False
        
        # Simula registrazione costo
        self.cost_manager.record_api_cost(
            session_id=session_id,
            api_type="chat_completion",
            model="gpt-4o-2024-11-20",
            provider="openai",
            tokens_input=1000,
            tokens_output=400,
            actual_cost=cost1,
            purpose="test_normal"
        )
        
        # Test aggiunta che supera budget
        allowed, result = self.cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=15000,
            tokens_output=8000,
            model="gpt-4o-2024-11-20"
        )
        
        cost2 = result['estimated_cost']
        status2 = result['session_budget']['budget_status']
        
        print(f"⚠️  Over budget attempt: ${cost2:.6f} - Status: {status2} - Allowed: {allowed}")
        
        if allowed or status2 != BudgetStatus.OVER_BUDGET:
            print("❌ Over budget usage should be blocked")
            return False
        
        print("✅ Session budget control: PASSED")
        return True
    
    def test_daily_budget_control(self) -> bool:
        """Test 3: Controllo budget giornaliero"""
        print("\n" + "="*60)
        print("📅 TEST 3: DAILY BUDGET CONTROL")
        print("="*60)
        
        # Simula più sessioni che si avvicinano al limite giornaliero
        session_costs = []
        
        for i in range(5):
            session_id = f"daily_test_{i}"
            
            # Prima sessione leggera, poi sempre più pesanti
            tokens_input = 1000 + (i * 1000)
            tokens_output = 400 + (i * 400)
            
            allowed, result = self.cost_manager.pre_api_check(
                session_id=session_id,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                model="gpt-4o-2024-11-20"
            )
            
            cost = result['estimated_cost']
            daily_status = result['daily_tracking']['daily_budget_status']
            daily_cost = result['daily_tracking']['current_daily_cost']
            
            print(f"Session {i+1}: ${cost:.6f} - Daily total: ${daily_cost:.4f} - Status: {daily_status} - Allowed: {allowed}")
            
            if allowed:
                # Registra solo se permesso
                self.cost_manager.record_api_cost(
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
                print(f"🚫 Session {i+1} blocked - Daily budget protection working")
                break
        
        # Verifica che abbiamo bloccato qualcosa prima di superare il limite
        total_cost = sum(session_costs)
        print(f"\n📊 Total recorded cost: ${total_cost:.4f} / ${self.cost_manager.max_daily_cost:.2f}")
        
        if total_cost >= self.cost_manager.max_daily_cost:
            print("❌ Daily budget control failed - exceeded limit")
            return False
        
        print("✅ Daily budget control: PASSED")
        return True
    
    def test_cost_spike_detection(self) -> bool:
        """Test 4: Rilevamento spike costi"""
        print("\n" + "="*60)
        print("📈 TEST 4: COST SPIKE DETECTION")
        print("="*60)
        
        # Crea baseline con sessioni normali (simulando giorni precedenti)
        import sqlite3
        from datetime import timedelta
        
        # Simula costi storici
        baseline_cost = 0.015  # $0.015 per sessione normale
        
        with sqlite3.connect(self.cost_manager.db_path) as conn:
            for days_ago in range(7, 1, -1):
                past_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                for session_num in range(3):  # 3 sessioni per giorno
                    session_id = f"baseline_{days_ago}_{session_num}"
                    
                    conn.execute("""
                        INSERT INTO api_costs 
                        (session_id, timestamp, api_type, model, provider, 
                         tokens_input, tokens_output, cost_usd, purpose)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id, past_date, "chat_completion", "gpt-4o-2024-11-20", "openai",
                        1500, 600, baseline_cost, "baseline_test"
                    ))
            conn.commit()
        
        print(f"📊 Created baseline: ${baseline_cost:.4f} average per session")
        
        # Test sessione normale (non dovrebbe triggerare spike)
        session_id = "spike_test_normal"
        allowed, result = self.cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=1500,
            tokens_output=600,
            model="gpt-4o-2024-11-20"
        )
        
        normal_spike = result['cost_alerts']['unusual_cost_spike']
        print(f"✅ Normal session spike detection: {normal_spike} (should be False)")
        
        # Test sessione anomala (dovrebbe triggerare spike)
        session_id = "spike_test_anomaly"
        allowed, result = self.cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=15000,  # 10x normale
            tokens_output=6000,
            model="gpt-4o-2024-11-20"
        )
        
        anomaly_spike = result['cost_alerts']['unusual_cost_spike']
        spike_cost = result['estimated_cost']
        print(f"🚨 Anomaly session spike detection: {anomaly_spike} (should be True)")
        print(f"   Cost: ${spike_cost:.6f} vs baseline ${baseline_cost:.6f}")
        
        if normal_spike or not anomaly_spike:
            print("❌ Cost spike detection not working correctly")
            return False
        
        print("✅ Cost spike detection: PASSED")
        return True
    
    def test_optimization_suggestions(self) -> bool:
        """Test 5: Suggerimenti ottimizzazione"""
        print("\n" + "="*60)
        print("💡 TEST 5: OPTIMIZATION SUGGESTIONS")
        print("="*60)
        
        # Test scenari che dovrebbero generare suggerimenti
        test_scenarios = [
            {
                "name": "High session cost",
                "session_id": "opt_test_expensive",
                "tokens_input": 8000,
                "tokens_output": 3000,
                "expected_suggestions": ["token_reduction"]
            },
            {
                "name": "Multiple daily sessions",
                "session_id": "opt_test_multiple",
                "tokens_input": 2000,
                "tokens_output": 800,
                "expected_suggestions": ["batch_processing"]
            }
        ]
        
        all_passed = True
        
        for scenario in test_scenarios:
            # Simula più sessioni se necessario
            if "multiple" in scenario["name"]:
                # Crea 6 sessioni oggi per triggerare suggestion
                for i in range(6):
                    fake_session = f"multiple_session_{i}"
                    self.cost_manager.record_api_cost(
                        session_id=fake_session,
                        api_type="chat_completion",
                        model="gpt-4o-2024-11-20",
                        provider="openai",
                        tokens_input=1000,
                        tokens_output=400,
                        actual_cost=0.01,
                        purpose="multiple_test"
                    )
            
            allowed, result = self.cost_manager.pre_api_check(
                session_id=scenario["session_id"],
                tokens_input=scenario["tokens_input"],
                tokens_output=scenario["tokens_output"],
                model="gpt-4o-2024-11-20"
            )
            
            suggestions = result['optimization_suggestions']
            suggestion_types = [s['type'] for s in suggestions]
            
            print(f"🧪 {scenario['name']}:")
            print(f"   Suggestions generated: {suggestion_types}")
            print(f"   Expected: {scenario['expected_suggestions']}")
            
            # Verifica che almeno un suggerimento atteso sia presente
            has_expected = any(exp in suggestion_types for exp in scenario['expected_suggestions'])
            
            if not has_expected:
                print(f"   ❌ Missing expected suggestions")
                all_passed = False
            else:
                print(f"   ✅ Suggestions appropriate")
        
        print(f"\n💡 Optimization suggestions test: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    def test_cli_integration(self) -> bool:
        """Test 6: Integrazione CLI"""
        print("\n" + "="*60)
        print("🖥️  TEST 6: CLI INTEGRATION")
        print("="*60)
        
        try:
            from cbt_journal.utils.cost_cli import CostControlCLI
            
            # Inizializza CLI con nostro cost manager
            cli = CostControlCLI()
            cli.cost_manager = self.cost_manager  # Override per usare test database
            
            print("✅ CLI import successful")
            
            # Test estimate command
            print("🧮 Testing estimate command...")
            cli.estimate_cost(2000, 800, "gpt-4o-2024-11-20")
            
            # Test summary command
            print("\n📊 Testing summary command...")
            cli.show_summary("today")
            
            # Test scenario command
            print("\n🧪 Testing scenario command...")
            cli.test_scenario("normal")
            
            print("\n✅ CLI integration: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ CLI integration failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_realistic_cbt_workflow(self) -> bool:
        """Test 7: Workflow CBT realistico"""
        print("\n" + "="*60)
        print("🧠 TEST 7: REALISTIC CBT WORKFLOW")
        print("="*60)
        
        # Simula giornata tipica di uso CBT
        workflow_steps = [
            {
                "step": "Morning reflection",
                "tokens_input": 800,
                "tokens_output": 600,
                "description": "Daily mood check and planning"
            },
            {
                "step": "Context retrieval",
                "tokens_input": 3000,  # Include historical context
                "tokens_output": 1200,
                "description": "Accessing historical patterns"
            },
            {
                "step": "Problem analysis",
                "tokens_input": 2500,
                "tokens_output": 1800,
                "description": "Deep dive into specific issue"
            },
            {
                "step": "Evening summary",
                "tokens_input": 1000,
                "tokens_output": 800,
                "description": "Day wrap-up and insights"
            }
        ]
        
        total_workflow_cost = 0
        session_id = "cbt_workflow_test"
        
        print("🔄 Simulating daily CBT workflow...")
        
        for i, step in enumerate(workflow_steps, 1):
            print(f"\n📝 Step {i}: {step['step']}")
            print(f"   Description: {step['description']}")
            
            allowed, result = self.cost_manager.pre_api_check(
                session_id=f"{session_id}_{i}",
                tokens_input=step['tokens_input'],
                tokens_output=step['tokens_output'],
                model="gpt-4o-2024-11-20"
            )
            
            step_cost = result['estimated_cost']
            total_workflow_cost += step_cost
            
            print(f"   Cost: ${step_cost:.6f}")
            print(f"   Allowed: {allowed}")
            
            if allowed:
                # Registra il costo
                self.cost_manager.record_api_cost(
                    session_id=f"{session_id}_{i}",
                    api_type="chat_completion",
                    model="gpt-4o-2024-11-20",
                    provider="openai",
                    tokens_input=step['tokens_input'],
                    tokens_output=step['tokens_output'],
                    actual_cost=step_cost,
                    purpose=step['step'].lower().replace(' ', '_')
                )
                print(f"   ✅ Step completed")
            else:
                print(f"   🚫 Step blocked by budget control")
                break
        
        # Analisi finale
        daily_summary = self.cost_manager.get_cost_summary("today")
        
        print(f"\n📊 WORKFLOW SUMMARY:")
        print(f"   Total workflow cost: ${total_workflow_cost:.4f}")
        print(f"   Daily total: ${daily_summary['total_cost']:.4f}")
        print(f"   Daily budget: ${daily_summary['budget_limit']:.2f}")
        print(f"   Utilization: {daily_summary['utilization']:.1%}")
        
        # Workflow dovrebbe essere completabile in budget normale
        workflow_affordable = total_workflow_cost <= (self.cost_manager.max_daily_cost * 0.8)
        
        if not workflow_affordable:
            print("❌ Realistic CBT workflow too expensive for daily budget")
            return False
        
        print("✅ Realistic CBT workflow: PASSED")
        return True
    
    def test_data_persistence(self) -> bool:
        """Test 8: Persistenza dati"""
        print("\n" + "="*60)
        print("💾 TEST 8: DATA PERSISTENCE")
        print("="*60)
        
        # Registra alcuni costi
        test_session = "persistence_test"
        
        self.cost_manager.record_api_cost(
            session_id=test_session,
            api_type="chat_completion",
            model="gpt-4o-2024-11-20",
            provider="openai",
            tokens_input=1000,
            tokens_output=500,
            actual_cost=0.0075,
            purpose="persistence_test"
        )
        
        # Crea nuovo manager con stesso database
        new_manager = CostControlManager(self.db_path)
        
        # Verifica che i dati persistano
        session_cost = new_manager._get_session_cost(test_session)
        daily_cost = new_manager._get_daily_cost(date.today().isoformat())
        
        print(f"📊 Session cost from new manager: ${session_cost:.6f}")
        print(f"📅 Daily cost from new manager: ${daily_cost:.6f}")
        
        if session_cost != 0.0075:
            print("❌ Session cost not persisted correctly")
            return False
        
        if daily_cost < 0.0075:
            print("❌ Daily cost not persisted correctly")
            return False
        
        print("✅ Data persistence: PASSED")
        return True
    
    def test_edge_cases(self) -> bool:
        """Test 9: Casi limite"""
        print("\n" + "="*60)
        print("⚠️  TEST 9: EDGE CASES")
        print("="*60)
        
        edge_cases = [
            {
                "name": "Zero tokens",
                "tokens_input": 0,
                "tokens_output": 0,
                "should_work": True
            },
            {
                "name": "Very large input",
                "tokens_input": 100000,
                "tokens_output": 10000,
                "should_work": True  # Dovrebbe funzionare ma essere bloccato da budget
            },
            {
                "name": "Only input tokens",
                "tokens_input": 1000,
                "tokens_output": 0,
                "should_work": True
            },
            {
                "name": "Only output tokens",
                "tokens_input": 0,
                "tokens_output": 1000,
                "should_work": True
            }
        ]
        
        all_passed = True
        
        for case in edge_cases:
            try:
                print(f"\n🧪 Testing: {case['name']}")
                
                cost = self.cost_manager.estimate_api_cost(
                    case['tokens_input'],
                    case['tokens_output'],
                    "gpt-4o-2024-11-20"
                )
                
                allowed, result = self.cost_manager.pre_api_check(
                    session_id=f"edge_case_{case['name'].replace(' ', '_')}",
                    tokens_input=case['tokens_input'],
                    tokens_output=case['tokens_output'],
                    model="gpt-4o-2024-11-20"
                )
                
                print(f"   Cost: ${cost:.6f}")
                print(f"   Allowed: {allowed}")
                print(f"   ✅ Edge case handled correctly")
                
            except Exception as e:
                print(f"   ❌ Edge case failed: {str(e)}")
                if case['should_work']:
                    all_passed = False
        
        print(f"\n⚠️  Edge cases test: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    def test_performance(self) -> bool:
        """Test 10: Performance"""
        print("\n" + "="*60)
        print("⚡ TEST 10: PERFORMANCE")
        print("="*60)
        
        import time
        
        # Test performance check budget
        start_time = time.time()
        
        for i in range(10):
            allowed, result = self.cost_manager.pre_api_check(
                session_id=f"perf_test_{i}",
                tokens_input=2000,
                tokens_output=800,
                model="gpt-4o-2024-11-20"
            )
        
        check_time = time.time() - start_time
        avg_check_time = check_time / 10
        
        print(f"📊 Budget check performance:")
        print(f"   10 checks in {check_time:.3f}s")
        print(f"   Average: {avg_check_time:.3f}s per check")
        
        # Test performance record cost
        start_time = time.time()
        
        for i in range(10):
            self.cost_manager.record_api_cost(
                session_id=f"perf_record_{i}",
                api_type="chat_completion",
                model="gpt-4o-2024-11-20",
                provider="openai",
                tokens_input=2000,
                tokens_output=800,
                actual_cost=0.015,
                purpose="performance_test"
            )
        
        record_time = time.time() - start_time
        avg_record_time = record_time / 10
        
        print(f"📊 Cost recording performance:")
        print(f"   10 records in {record_time:.3f}s")
        print(f"   Average: {avg_record_time:.3f}s per record")
        
        # Performance dovrebbe essere < 0.1s per operazione
        if avg_check_time > 0.1 or avg_record_time > 0.1:
            print("❌ Performance too slow for real-time usage")
            return False
        
        print("✅ Performance: PASSED")
        return True
    
    def run_all_tests(self) -> bool:
        """Esegui tutti i test"""
        print("🚀 STARTING COMPLETE COST CONTROL TEST SUITE")
        print("=" * 70)
        
        tests = [
            ("Cost Estimation", self.test_cost_estimation),
            ("Session Budget Control", self.test_session_budget_control),
            ("Daily Budget Control", self.test_daily_budget_control),
            ("Cost Spike Detection", self.test_cost_spike_detection),
            ("Optimization Suggestions", self.test_optimization_suggestions),
            ("CLI Integration", self.test_cli_integration),
            ("Realistic CBT Workflow", self.test_realistic_cbt_workflow),
            ("Data Persistence", self.test_data_persistence),
            ("Edge Cases", self.test_edge_cases),
            ("Performance", self.test_performance)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                
            except Exception as e:
                print(f"\n❌ {test_name} CRASHED: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary risultati
        print("\n" + "="*70)
        print("📋 TEST SUITE SUMMARY")
        print("="*70)
        
        passed_count = 0
        total_count = len(results)
        
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} {test_name}")
            if passed:
                passed_count += 1
        
        success_rate = (passed_count / total_count) * 100
        
        print("\n" + "-"*70)
        print(f"📊 RESULTS: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED - COST CONTROL SYSTEM READY!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW ISSUES BEFORE DEPLOYMENT")
            return False


def main():
    """Main test function"""
    
    test_suite = CostControlTestSuite()
    
    try:
        success = test_suite.run_all_tests()
        return 0 if success else 1
        
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    exit(main())
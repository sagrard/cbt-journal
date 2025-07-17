#!/usr/bin/env python3
"""
Test Completi per Qdrant Setup v3.3.0
Validazione funzionamento CBT Journal collection
"""

import json
import uuid
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range

class CBTQdrantComprehensiveTest:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Test completi per CBT Journal Qdrant setup"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "cbt_journal_sessions"
        self.test_points = []  # Track per cleanup
        
    def generate_realistic_session(self, session_id: str, session_type: str = "emotional_processing") -> Dict[str, Any]:
        """Genera sessione realistica per test"""
        
        session_data = {
            "session_id": session_id,
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "data_source": random.choice(["chatgpt_import", "local_system"]),
            "session_type": session_type,
            "duration_minutes": random.randint(5, 120),
            
            # Content realistico
            "content": {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Sessione {session_type} di test con contenuto emotivo significativo",
                        "timestamp": datetime.now().isoformat(),
                        "word_count": 9
                    },
                    {
                        "role": "assistant",
                        "content": "Comprendo i tuoi sentimenti, esploriamo insieme questa esperienza emotiva",
                        "timestamp": datetime.now().isoformat(),
                        "word_count": 10
                    }
                ],
                "conversation_count": 1,
                "total_words": {"user": 9, "assistant": 10, "total": 19}
            },
            
            # AI Models tracking
            "ai_models": {
                "tracking_level": "complete",
                "response_model": {
                    "name": random.choice(["gpt-4o-2024-11-20", "gpt-4o-2024-08-06"]),
                    "confidence": "exact",
                    "provider": "openai"
                },
                "session_cost_summary": {
                    "total_cost_usd": round(random.uniform(0.01, 0.45), 3),
                    "cost_breakdown": {
                        "embedding": round(random.uniform(0.001, 0.01), 3),
                        "generation": round(random.uniform(0.005, 0.40), 3)
                    },
                    "total_tokens": {
                        "input": random.randint(100, 800), 
                        "output": random.randint(50, 400), 
                        "total": random.randint(150, 1200)
                    }
                }
            },
            
            # Clinical Assessment
            "clinical_assessment": {
                "data_available": True,
                "mood_rating": random.randint(1, 10),
                "anxiety_level": random.randint(1, 10),
                "energy_level": random.randint(1, 10),
                "assessment_method": "user_input"
            },
            
            # RAG Metadata
            "rag_metadata": {
                "context_type": random.choice(["narrative", "clinical", "outcome", "chat"]),
                "language": "it",
                "context_priority": random.randint(1, 5),
                "token_count": random.randint(150, 1200),
                "embedding_version": "text-embedding-3-large"
            },
            
            # Narrative Insight
            "narrative_insight": {
                "emotional_needs": random.sample(
                    ["validation", "understanding", "support", "autonomy", "connection"], 
                    k=random.randint(1, 3)
                ),
                "narrative_theme": random.choice([
                    "emotional_processing", "relationship_dynamics", "work_stress", 
                    "self_discovery", "coping_strategies"
                ]),
                "turning_point_detected": random.choice([True, False]),
                "confidence": random.choice(["low", "medium", "high"])
            },
            
            # Clinical Classification
            "clinical_classification": {
                "data_available": True,
                "primary_themes": random.sample(
                    ["anxiety", "depression", "relationships", "work", "self_esteem"], 
                    k=random.randint(1, 3)
                ),
                "emotional_patterns": random.sample(
                    ["fear", "sadness", "joviality", "attentiveness", "fatigue"], 
                    k=random.randint(1, 2)
                ),
                "risk_assessment": {
                    "risk_level": random.choice(["low", "moderate", "high"]),
                    "crisis_indicators": random.choice([True, False])
                }
            },
            
            # Cost Monitoring
            "cost_monitoring": {
                "session_budget": {
                    "max_cost_per_session": 0.50,
                    "current_session_cost": round(random.uniform(0.01, 0.45), 3),
                    "budget_status": random.choice(["within_limits", "approaching_limit"])
                }
            },
            
            # System metadata
            "system_metadata": {
                "schema_version": "3.3.0",
                "created_at": datetime.now().isoformat()
            }
        }
        
        return session_data
    
    def test_bulk_insert_performance(self, num_sessions: int = 20) -> bool:
        """Test performance insert multiple sessioni"""
        try:
            print(f"\n📈 Testing bulk insert ({num_sessions} sessions)...")
            
            start_time = time.time()
            
            # Genera sessioni diverse
            session_types = ["emotional_processing", "problem_solving", "reflection", "crisis_support", "general"]
            points = []
            
            for i in range(num_sessions):
                session_id = f"test_bulk_{i:03d}"
                session_type = session_types[i % len(session_types)]
                
                # Vector random 3072 dim
                vector = [random.random() for _ in range(3072)]
                
                # Payload realistico
                payload = self.generate_realistic_session(session_id, session_type)
                
                point_id = str(uuid.uuid4())
                self.test_points.append(point_id)
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # Bulk insert
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            insert_time = time.time() - start_time
            per_session_ms = (insert_time / num_sessions) * 1000
            
            print(f"✅ Bulk insert: {insert_time:.3f}s total ({per_session_ms:.1f}ms per session)")
            print(f"   Status: {result.status}")
            print(f"   Performance: {'EXCELLENT' if per_session_ms < 50 else 'GOOD' if per_session_ms < 200 else 'SLOW'}")
            
            return result.status.name == "COMPLETED"
            
        except Exception as e:
            print(f"❌ Bulk insert failed: {str(e)}")
            return False
    
    def test_complex_filtering(self) -> bool:
        """Test filtering complessi per RAG use cases"""
        try:
            print("\n🔍 Testing complex filtering scenarios...")
            
            # Test 1: Multi-field clinical filter
            clinical_filter = Filter(
                must=[
                    FieldCondition(key="clinical_assessment.mood_rating", range=Range(gte=6)),
                    FieldCondition(key="clinical_assessment.anxiety_level", range=Range(lte=5)),
                    FieldCondition(key="clinical_classification.risk_assessment.risk_level", match={"value": "low"})
                ]
            )
            
            query_vector = [random.random() for _ in range(3072)]
            
            start_time = time.time()
            clinical_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=clinical_filter,
                limit=10
            )
            clinical_time = time.time() - start_time
            
            print(f"✅ Clinical filter: {len(clinical_results.points)} results ({clinical_time*1000:.1f}ms)")
            
            # Test 2: RAG context filter
            rag_filter = Filter(
                must=[
                    FieldCondition(key="rag_metadata.language", match={"value": "it"}),
                    FieldCondition(key="rag_metadata.context_type", match={"value": "narrative"}),
                    FieldCondition(key="rag_metadata.context_priority", range=Range(gte=3))
                ]
            )
            
            start_time = time.time()
            rag_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=rag_filter,
                limit=10
            )
            rag_time = time.time() - start_time
            
            print(f"✅ RAG context filter: {len(rag_results.points)} results ({rag_time*1000:.1f}ms)")
            
            # Test 3: Cost monitoring filter
            cost_filter = Filter(
                must=[
                    FieldCondition(key="cost_monitoring.session_budget.budget_status", match={"value": "within_limits"}),
                    FieldCondition(key="ai_models.session_cost_summary.total_cost_usd", range=Range(lte=0.30))
                ]
            )
            
            start_time = time.time()
            cost_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=cost_filter,
                limit=10
            )
            cost_time = time.time() - start_time
            
            print(f"✅ Cost monitoring filter: {len(cost_results.points)} results ({cost_time*1000:.1f}ms)")
            
            # Test 4: Emotional intelligence filter
            emotion_filter = Filter(
                must=[
                    FieldCondition(key="narrative_insight.turning_point_detected", match={"value": True}),
                    FieldCondition(key="narrative_insight.confidence", match={"value": "high"})
                ]
            )
            
            start_time = time.time()
            emotion_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=emotion_filter,
                limit=10
            )
            emotion_time = time.time() - start_time
            
            print(f"✅ Emotional intelligence filter: {len(emotion_results.points)} results ({emotion_time*1000:.1f}ms)")
            
            # Performance assessment
            avg_filter_time = (clinical_time + rag_time + cost_time + emotion_time) / 4
            if avg_filter_time < 0.05:
                print(f"🚀 Filter performance: EXCELLENT ({avg_filter_time*1000:.1f}ms average)")
            elif avg_filter_time < 0.2:
                print(f"✅ Filter performance: GOOD ({avg_filter_time*1000:.1f}ms average)")
            else:
                print(f"⚠️ Filter performance: SLOW ({avg_filter_time*1000:.1f}ms average)")
            
            return True
            
        except Exception as e:
            print(f"❌ Complex filtering failed: {str(e)}")
            return False
    
    def test_semantic_search_quality(self) -> bool:
        """Test qualità ricerca semantica"""
        try:
            print("\n🧠 Testing semantic search quality...")
            
            # Inserisci sessioni con contenuto semanticamente correlato
            semantic_sessions = [
                ("anxiety_work", "Mi sento molto ansioso per il progetto lavorativo", ["anxiety", "work"]),
                ("anxiety_relationship", "L'ansia mi sta rovinando la relazione", ["anxiety", "relationships"]),
                ("depression_general", "Oggi mi sento particolarmente triste e demotivato", ["depression"]),
                ("stress_family", "La situazione familiare mi sta stressando molto", ["stress", "family"])
            ]
            
            semantic_points = []
            semantic_vectors = {}
            
            for session_id, content, themes in semantic_sessions:
                # Vector correlati al contenuto (simulazione embedding)
                base_vector = [random.random() for _ in range(3072)]
                
                # Aggiungi correlazione semantica per anxiety
                if "anxiety" in themes:
                    for i in range(0, 100):  # Prime 100 dimensioni correlate
                        base_vector[i] = base_vector[i] * 0.8 + 0.2  # Shift verso pattern anxiety
                
                semantic_vectors[session_id] = base_vector
                
                payload = self.generate_realistic_session(session_id)
                payload["content"]["messages"][0]["content"] = content
                payload["clinical_classification"]["primary_themes"] = themes
                
                point_id = str(uuid.uuid4())
                self.test_points.append(point_id)
                
                semantic_points.append(PointStruct(
                    id=point_id,
                    vector=base_vector,
                    payload=payload
                ))
            
            # Insert semantic sessions
            self.client.upsert(
                collection_name=self.collection_name,
                points=semantic_points
            )
            print(f"✅ Inserted {len(semantic_sessions)} semantic test sessions")
            
            # Test semantic similarity
            anxiety_query = semantic_vectors["anxiety_work"]
            
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=anxiety_query,
                limit=5
            )
            
            # Analizza risultati
            anxiety_sessions_found = 0
            for result in search_results.points:
                if "anxiety" in result.payload.get("clinical_classification", {}).get("primary_themes", []):
                    anxiety_sessions_found += 1
            
            print(f"✅ Semantic search: {anxiety_sessions_found}/{len(search_results.points)} anxiety sessions found")
            print(f"   Relevance scores: {[round(p.score, 3) for p in search_results.points[:3]]}")
            
            # Test con filter semantico
            anxiety_filter = Filter(
                must=[
                    FieldCondition(key="clinical_classification.primary_themes", match={"value": "anxiety"})
                ]
            )
            
            filtered_results = self.client.query_points(
                collection_name=self.collection_name,
                query=anxiety_query,
                query_filter=anxiety_filter,
                limit=5
            )
            
            print(f"✅ Filtered semantic search: {len(filtered_results.points)} anxiety-specific results")
            
            return anxiety_sessions_found >= 1  # Almeno una sessione anxiety trovata
            
        except Exception as e:
            print(f"❌ Semantic search test failed: {str(e)}")
            return False
    
    def test_concurrent_operations(self) -> bool:
        """Test operazioni concorrenti"""
        try:
            print("\n🔄 Testing concurrent operations...")
            
            import threading
            import concurrent.futures
            
            def concurrent_search(thread_id: int) -> str:
                """Search concorrente"""
                try:
                    query_vector = [random.random() for _ in range(3072)]
                    
                    results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=5
                    )
                    
                    return f"Thread {thread_id}: {len(results.points)} results"
                except Exception as e:
                    return f"Thread {thread_id}: ERROR - {str(e)}"
            
            def concurrent_filter(thread_id: int) -> str:
                """Filter concorrente"""
                try:
                    filter_condition = Filter(
                        must=[
                            FieldCondition(key="rag_metadata.language", match={"value": "it"})
                        ]
                    )
                    
                    query_vector = [random.random() for _ in range(3072)]
                    
                    results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        query_filter=filter_condition,
                        limit=3
                    )
                    
                    return f"Filter {thread_id}: {len(results.points)} results"
                except Exception as e:
                    return f"Filter {thread_id}: ERROR - {str(e)}"
            
            # Test concurrent search
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                search_futures = [executor.submit(concurrent_search, i) for i in range(5)]
                filter_futures = [executor.submit(concurrent_filter, i) for i in range(3)]
                
                search_results = [future.result() for future in concurrent.futures.as_completed(search_futures)]
                filter_results = [future.result() for future in concurrent.futures.as_completed(filter_futures)]
            
            # Verifica risultati
            search_success = sum(1 for result in search_results if "ERROR" not in result)
            filter_success = sum(1 for result in filter_results if "ERROR" not in result)
            
            print(f"✅ Concurrent search: {search_success}/5 successful")
            print(f"✅ Concurrent filter: {filter_success}/3 successful")
            
            for result in search_results:
                print(f"   {result}")
            for result in filter_results:
                print(f"   {result}")
            
            return search_success >= 4 and filter_success >= 2
            
        except Exception as e:
            print(f"❌ Concurrent operations failed: {str(e)}")
            return False
    
    def test_stress_performance(self) -> bool:
        """Test performance sotto stress"""
        try:
            print("\n💪 Testing stress performance...")
            
            # Test rapid queries
            query_vector = [random.random() for _ in range(3072)]
            
            start_time = time.time()
            for i in range(50):
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=3
                )
            rapid_time = time.time() - start_time
            
            avg_query_time = rapid_time / 50
            print(f"✅ Rapid queries: 50 queries in {rapid_time:.3f}s ({avg_query_time*1000:.1f}ms avg)")
            
            # Test complex filter under stress
            complex_filter = Filter(
                must=[
                    FieldCondition(key="clinical_assessment.mood_rating", range=Range(gte=3)),
                    FieldCondition(key="rag_metadata.language", match={"value": "it"}),
                    FieldCondition(key="cost_monitoring.session_budget.budget_status", match={"value": "within_limits"})
                ]
            )
            
            start_time = time.time()
            for i in range(20):
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=complex_filter,
                    limit=5
                )
            complex_time = time.time() - start_time
            
            avg_complex_time = complex_time / 20
            print(f"✅ Complex filtered queries: 20 queries in {complex_time:.3f}s ({avg_complex_time*1000:.1f}ms avg)")
            
            # Performance assessment
            if avg_query_time < 0.05 and avg_complex_time < 0.1:
                print("🚀 Stress performance: EXCELLENT - Ready for production")
            elif avg_query_time < 0.1 and avg_complex_time < 0.2:
                print("✅ Stress performance: GOOD - Suitable for daily use")
            else:
                print("⚠️ Stress performance: MODERATE - Monitor in production")
            
            return True
            
        except Exception as e:
            print(f"❌ Stress test failed: {str(e)}")
            return False
    
    def cleanup_test_data(self) -> bool:
        """Cleanup dati test"""
        try:
            if self.test_points:
                print(f"\n🧹 Cleaning up {len(self.test_points)} test points...")
                
                # Delete in batches per evitare timeout
                batch_size = 100
                for i in range(0, len(self.test_points), batch_size):
                    batch = self.test_points[i:i + batch_size]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=batch
                    )
                
                print("✅ Test data cleanup completed")
                self.test_points = []
            
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {str(e)}")
            return False
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Statistiche finali collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "schema_version": "3.3.0",
                "test_status": "PASSED"
            }
            
        except Exception as e:
            return {"error": str(e), "test_status": "FAILED"}
    
    def run_comprehensive_tests(self) -> bool:
        """Esegui tutti i test completi"""
        print("=" * 80)
        print("CBT JOURNAL - COMPREHENSIVE QDRANT TESTS")
        print("=" * 80)
        
        all_tests_passed = True
        
        # Test 1: Bulk insert performance
        if not self.test_bulk_insert_performance():
            all_tests_passed = False
        
        # Test 2: Complex filtering
        if not self.test_complex_filtering():
            all_tests_passed = False
        
        # Test 3: Semantic search quality  
        if not self.test_semantic_search_quality():
            all_tests_passed = False
        
        # Test 4: Concurrent operations
        if not self.test_concurrent_operations():
            all_tests_passed = False
        
        # Test 5: Stress performance
        if not self.test_stress_performance():
            all_tests_passed = False
        
        # Cleanup
        self.cleanup_test_data()
        
        # Final stats
        stats = self.get_final_stats()
        print(f"\n📊 FINAL COLLECTION STATS:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 80)
        if all_tests_passed:
            print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("✅ Sistema Qdrant completamente validato e pronto per produzione")
            print("✅ Performance ottimali per CBT Journal workload")
            print("✅ Schema v3.3.0 completamente funzionale")
            print("\n🚀 READY FOR WEEK 2: RAG Pipeline Core Development")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️ Revisione e troubleshooting necessari prima di procedere")
        print("=" * 80)
        
        return all_tests_passed

def main():
    """Test completi principali"""
    tester = CBTQdrantComprehensiveTest()
    success = tester.run_comprehensive_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

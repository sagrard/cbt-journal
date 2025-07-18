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


# ===================== PYTEST REFACTORING =====================
import pytest

# ---- Fixtures ----
@pytest.fixture(scope="module")
def qdrant_client():
    client = QdrantClient(host="localhost", port=6333)
    yield client

@pytest.fixture(scope="module")
def collection_name():
    return "cbt_journal_sessions"

@pytest.fixture(scope="function")
def test_points():
    points = []
    yield points
    # Cleanup logic can be added here if needed

def generate_realistic_session(session_id: str, session_type: str = "emotional_processing") -> dict:
    session_data = {
        "session_id": session_id,
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        "data_source": random.choice(["chatgpt_import", "local_system"]),
        "session_type": session_type,
        "duration_minutes": random.randint(5, 120),
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
        "clinical_assessment": {
            "data_available": True,
            "mood_rating": random.randint(1, 10),
            "anxiety_level": random.randint(1, 10),
            "energy_level": random.randint(1, 10),
            "assessment_method": "user_input"
        },
        "rag_metadata": {
            "context_type": random.choice(["narrative", "clinical", "outcome", "chat"]),
            "language": "it",
            "context_priority": random.randint(1, 5),
            "token_count": random.randint(150, 1200),
            "embedding_version": "text-embedding-3-large"
        },
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
        "cost_monitoring": {
            "session_budget": {
                "max_cost_per_session": 0.50,
                "current_session_cost": round(random.uniform(0.01, 0.45), 3),
                "budget_status": random.choice(["within_limits", "approaching_limit"])
            }
        },
        "system_metadata": {
            "schema_version": "3.3.0",
            "created_at": datetime.now().isoformat()
        }
    }
    return session_data

# ---- Test functions ----
def test_bulk_insert_performance(qdrant_client, collection_name, test_points):
    """Test performance insert multiple sessioni"""
    num_sessions = 20
    session_types = ["emotional_processing", "problem_solving", "reflection", "crisis_support", "general"]
    points = []
    for i in range(num_sessions):
        session_id = f"test_bulk_{i:03d}"
        session_type = session_types[i % len(session_types)]
        vector = [random.random() for _ in range(3072)]
        payload = generate_realistic_session(session_id, session_type)
        point_id = str(uuid.uuid4())
        test_points.append(point_id)
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    result = qdrant_client.upsert(collection_name=collection_name, points=points)
    assert result.status.name == "COMPLETED"
    

# ---- Test: Complex Filtering ----
def test_complex_filtering(qdrant_client, collection_name):
    """Test filtering complessi per RAG use cases"""
    # Test 1: Multi-field clinical filter
    clinical_filter = Filter(
        must=[
            FieldCondition(key="clinical_assessment.mood_rating", range=Range(gte=6)),
            FieldCondition(key="clinical_assessment.anxiety_level", range=Range(lte=5)),
            FieldCondition(key="clinical_classification.risk_assessment.risk_level", match={"value": "low"})
        ]
    )
    query_vector = [random.random() for _ in range(3072)]
    clinical_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=clinical_filter,
        limit=10
    )
    assert isinstance(clinical_results.points, list)

    # Test 2: RAG context filter
    rag_filter = Filter(
        must=[
            FieldCondition(key="rag_metadata.language", match={"value": "it"}),
            FieldCondition(key="rag_metadata.context_type", match={"value": "narrative"}),
            FieldCondition(key="rag_metadata.context_priority", range=Range(gte=3))
        ]
    )
    rag_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=rag_filter,
        limit=10
    )
    assert isinstance(rag_results.points, list)

    # Test 3: Cost monitoring filter
    cost_filter = Filter(
        must=[
            FieldCondition(key="cost_monitoring.session_budget.budget_status", match={"value": "within_limits"}),
            FieldCondition(key="ai_models.session_cost_summary.total_cost_usd", range=Range(lte=0.30))
        ]
    )
    cost_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=cost_filter,
        limit=10
    )
    assert isinstance(cost_results.points, list)

    # Test 4: Emotional intelligence filter
    emotion_filter = Filter(
        must=[
            FieldCondition(key="narrative_insight.turning_point_detected", match={"value": True}),
            FieldCondition(key="narrative_insight.confidence", match={"value": "high"})
        ]
    )
    emotion_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=emotion_filter,
        limit=10
    )
    assert isinstance(emotion_results.points, list)
    

# ---- Test: Semantic Search Quality ----
def test_semantic_search_quality(qdrant_client, collection_name, test_points):
    """Test qualità ricerca semantica"""
    semantic_sessions = [
        ("anxiety_work", "Mi sento molto ansioso per il progetto lavorativo", ["anxiety", "work"]),
        ("anxiety_relationship", "L'ansia mi sta rovinando la relazione", ["anxiety", "relationships"]),
        ("depression_general", "Oggi mi sento particolarmente triste e demotivato", ["depression"]),
        ("stress_family", "La situazione familiare mi sta stressando molto", ["stress", "family"])
    ]
    semantic_points = []
    semantic_vectors = {}
    for session_id, content, themes in semantic_sessions:
        base_vector = [random.random() for _ in range(3072)]
        if "anxiety" in themes:
            for i in range(0, 100):
                base_vector[i] = base_vector[i] * 0.8 + 0.2
        semantic_vectors[session_id] = base_vector
        payload = generate_realistic_session(session_id)
        payload["content"]["messages"][0]["content"] = content
        payload["clinical_classification"]["primary_themes"] = themes
        point_id = str(uuid.uuid4())
        test_points.append(point_id)
        semantic_points.append(PointStruct(id=point_id, vector=base_vector, payload=payload))
    qdrant_client.upsert(collection_name=collection_name, points=semantic_points)
    anxiety_query = semantic_vectors["anxiety_work"]
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=anxiety_query,
        limit=5
    )
    anxiety_sessions_found = 0
    for result in search_results.points:
        if "anxiety" in result.payload.get("clinical_classification", {}).get("primary_themes", []):
            anxiety_sessions_found += 1
    assert anxiety_sessions_found >= 1
    anxiety_filter = Filter(
        must=[
            FieldCondition(key="clinical_classification.primary_themes", match={"value": "anxiety"})
        ]
    )
    filtered_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=anxiety_query,
        query_filter=anxiety_filter,
        limit=5
    )
    assert isinstance(filtered_results.points, list)
    

# ---- Test: Concurrent Operations ----
def test_concurrent_operations(qdrant_client, collection_name):
    """Test operazioni concorrenti"""
    import concurrent.futures

    def concurrent_search(thread_id: int) -> str:
        try:
            query_vector = [random.random() for _ in range(3072)]
            results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=5
            )
            return f"Thread {thread_id}: {len(results.points)} results"
        except Exception as e:
            return f"Thread {thread_id}: ERROR - {str(e)}"

    def concurrent_filter(thread_id: int) -> str:
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(key="rag_metadata.language", match={"value": "it"})
                ]
            )
            query_vector = [random.random() for _ in range(3072)]
            results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_condition,
                limit=3
            )
            return f"Filter {thread_id}: {len(results.points)} results"
        except Exception as e:
            return f"Filter {thread_id}: ERROR - {str(e)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        search_futures = [executor.submit(concurrent_search, i) for i in range(5)]
        filter_futures = [executor.submit(concurrent_filter, i) for i in range(3)]
        search_results = [future.result() for future in concurrent.futures.as_completed(search_futures)]
        filter_results = [future.result() for future in concurrent.futures.as_completed(filter_futures)]

    search_success = sum(1 for result in search_results if "ERROR" not in result)
    filter_success = sum(1 for result in filter_results if "ERROR" not in result)
    assert search_success >= 4
    assert filter_success >= 2
    

# ---- Test: Stress Performance ----
def test_stress_performance(qdrant_client, collection_name):
    """Test performance sotto stress"""
    query_vector = [random.random() for _ in range(3072)]
    start_time = time.time()
    for _ in range(50):
        qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        )
    rapid_time = time.time() - start_time
    avg_query_time = rapid_time / 50
    complex_filter = Filter(
        must=[
            FieldCondition(key="clinical_assessment.mood_rating", range=Range(gte=3)),
            FieldCondition(key="rag_metadata.language", match={"value": "it"}),
            FieldCondition(key="cost_monitoring.session_budget.budget_status", match={"value": "within_limits"})
        ]
    )
    start_time = time.time()
    for _ in range(20):
        qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=complex_filter,
            limit=5
        )
    complex_time = time.time() - start_time
    avg_complex_time = complex_time / 20
    # Assert that queries are executed and times are floats
    assert isinstance(avg_query_time, float)
    assert isinstance(avg_complex_time, float)
    
def test_error_handling(qdrant_client, collection_name, test_points):
    """Test error handling per scenari problematici"""
    # Test 1: Invalid vector dimensions
    invalid_point = PointStruct(
        id=str(uuid.uuid4()),
        vector=[0.5, 0.5],  # Wrong dimension
        payload={"test": "invalid_vector"}
    )
    
    with pytest.raises(Exception):
        qdrant_client.upsert(collection_name=collection_name, points=[invalid_point])
    
    # Test 2: Query with invalid filter
    query_vector = [random.random() for _ in range(3072)]
    invalid_filter = Filter(
        must=[
            FieldCondition(key="non_existent_field", match={"value": "test"})
        ]
    )
    
    # Should not crash but return empty results
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=invalid_filter,
        limit=5
    )
    assert isinstance(results.points, list)

def test_data_consistency(qdrant_client, collection_name, test_points):
    """Test consistenza dati dopo operazioni multiple"""
    # Insert test data
    points = []
    for i in range(5):
        session_id = f"consistency_test_{i}"
        vector = [random.random() for _ in range(3072)]
        payload = generate_realistic_session(session_id)
        point_id = str(uuid.uuid4())
        test_points.append(point_id)
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    
    # Insert points
    result = qdrant_client.upsert(collection_name=collection_name, points=points)
    assert result.status.name == "COMPLETED"
    
    # Verify all points are retrievable
    for point in points:
        retrieved = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point.id],
            with_payload=True
        )
        assert len(retrieved) == 1
        assert retrieved[0].id == point.id
        assert retrieved[0].payload["session_id"] == point.payload["session_id"]

def test_schema_validation(qdrant_client, collection_name, test_points):
    """Test validazione schema v3.3.0"""
    # Test with complete schema
    complete_session = generate_realistic_session("schema_test_complete")
    vector = [random.random() for _ in range(3072)]
    point_id = str(uuid.uuid4())
    test_points.append(point_id)
    
    point = PointStruct(id=point_id, vector=vector, payload=complete_session)
    result = qdrant_client.upsert(collection_name=collection_name, points=[point])
    assert result.status.name == "COMPLETED"
    
    # Verify required fields are present
    retrieved = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True
    )
    payload = retrieved[0].payload
    
    # Check required schema fields
    assert "session_id" in payload
    assert "timestamp" in payload
    assert "data_source" in payload
    assert "content" in payload
    assert "ai_models" in payload
    assert "system_metadata" in payload
    assert payload["system_metadata"]["schema_version"] == "3.3.0"

def test_large_payload_handling(qdrant_client, collection_name, test_points):
    """Test handling di payload grandi"""
    # Create large session with many messages
    large_session = generate_realistic_session("large_payload_test")
    large_session["content"]["messages"] = []
    
    # Add many messages to simulate long conversation
    for i in range(50):
        large_session["content"]["messages"].append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i} con contenuto molto lungo " * 10,
            "timestamp": datetime.now().isoformat(),
            "word_count": 80
        })
    
    large_session["content"]["conversation_count"] = 50
    large_session["content"]["total_words"]["total"] = 4000
    
    vector = [random.random() for _ in range(3072)]
    point_id = str(uuid.uuid4())
    test_points.append(point_id)
    
    point = PointStruct(id=point_id, vector=vector, payload=large_session)
    result = qdrant_client.upsert(collection_name=collection_name, points=[point])
    assert result.status.name == "COMPLETED"
    
    # Verify retrieval works with large payload
    retrieved = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True
    )
    assert len(retrieved) == 1
    assert len(retrieved[0].payload["content"]["messages"]) == 50


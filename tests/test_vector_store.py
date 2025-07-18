#!/usr/bin/env python3
"""
Test Suite for CBT Vector Store
Comprehensive testing of all vector operations
"""

import os
import sys
import uuid
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from cbt_journal.rag.vector_store import CBTVectorStore, CBTVectorStoreError, create_vector_store
from cbt_journal.utils.cost_control import CostControlManager



# ===================== PYTEST REFACTORING =====================
import pytest

# ---- Fixtures ----

@pytest.fixture(scope="module")
def temp_dir():
    d = tempfile.mkdtemp(prefix="cbt_vector_test_")
    yield d
    shutil.rmtree(d)

@pytest.fixture
def collection_name():
    return "test_cbt_sessions"

@pytest.fixture
def mock_client(collection_name):
    client = Mock(spec=QdrantClient)
    # Setup collection verification
    mock_collection = Mock()
    mock_collection.name = collection_name
    mock_collections = Mock()
    mock_collections.collections = [mock_collection]
    client.get_collections.return_value = mock_collections
    # Setup get_collection
    mock_info = Mock()
    mock_info.config = Mock()
    mock_info.config.params = Mock()
    mock_info.config.params.vectors = Mock()
    mock_info.config.params.vectors.size = 3072
    client.get_collection.return_value = mock_info
    return client

@pytest.fixture
def cost_manager(temp_dir):
    return CostControlManager(db_path=os.path.join(temp_dir, "test_costs.db"))

@pytest.fixture
def vector_store(mock_client, cost_manager, collection_name):
    return CBTVectorStore(
        qdrant_client=mock_client,
        cost_manager=cost_manager,
        collection_name=collection_name
    )

# ---- Helper functions ----

def create_test_session(session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "data_source": "local_system",
        "session_type": "emotional_processing",
        "duration_minutes": 25,
        "content": {
            "messages": [
                {
                    "role": "user",
                    "content": "Test message for validation",
                    "timestamp": datetime.now().isoformat(),
                    "word_count": 5
                }
            ],
            "conversation_count": 1,
            "total_words": {"user": 5, "assistant": 0, "total": 5}
        },
        "ai_models": {
            "tracking_level": "complete",
            "response_model": {
                "name": "gpt-4o-2024-11-20",
                "confidence": "exact"
            }
        }
    }

def create_test_embedding():
    return [random.random() for _ in range(3072)]

# ---- Test functions ----

def test_initialization(mock_client, cost_manager, collection_name):
    # Should succeed
    store = CBTVectorStore(
        qdrant_client=mock_client,
        cost_manager=cost_manager,
        collection_name=collection_name
    )
    assert store.collection_name == collection_name
    # Should fail with wrong collection name
    mock_collections = Mock()
    mock_collections.collections = [Mock(name="different_collection")]
    with patch.object(mock_client, 'get_collections', return_value=mock_collections):
        with pytest.raises(CBTVectorStoreError):
            CBTVectorStore(
                qdrant_client=mock_client,
                cost_manager=cost_manager,
                collection_name="non_existent_collection"
            )

def test_store_session(vector_store, mock_client):
    session_data = create_test_session()
    embedding = create_test_embedding()
    # Mock upsert
    mock_result = Mock()
    mock_result.status.name = "COMPLETED"
    mock_client.upsert.return_value = mock_result
    session_id = vector_store.store_session(session_data, embedding)
    assert session_id == session_data["session_id"]
    mock_client.upsert.assert_called_once()
    # Invalid embedding
    with pytest.raises(CBTVectorStoreError):
        vector_store.store_session(session_data, [1, 2, 3])
    # Missing required fields
    with pytest.raises(CBTVectorStoreError):
        vector_store.store_session({"session_id": "test"}, embedding)

def test_search_similar(vector_store, mock_client):
    # Mock search results
    mock_point = Mock()
    mock_point.id = "test_session_123"
    mock_point.score = 0.85
    mock_point.payload = create_test_session("test_session_123")
    mock_results = Mock()
    mock_results.points = [mock_point]
    mock_client.query_points.return_value = mock_results
    query_embedding = create_test_embedding()
    results = vector_store.search_similar(query_embedding, limit=5)
    assert len(results) == 1
    result = results[0]
    assert result["session_id"] == "test_session_123"
    assert result["score"] == 0.85
    # Filters
    filters = {
        'data_source': 'local_system',
        'clinical_assessment.mood_rating': {'range': {'gte': 5, 'lte': 8}}
    }
    vector_store.search_similar(query_embedding, filters=filters, limit=10)
    # Invalid embedding
    with pytest.raises(CBTVectorStoreError):
        vector_store.search_similar([1, 2, 3], limit=5)

def test_get_session(vector_store, mock_client):
    session_id = "test_direct_retrieval"
    mock_point = Mock()
    mock_point.id = session_id
    mock_point.payload = create_test_session(session_id)
    mock_client.retrieve.return_value = [mock_point]
    result = vector_store.get_session(session_id)
    assert result is not None
    assert result["session_id"] == session_id
    # Non-existent session
    mock_client.retrieve.return_value = []
    result = vector_store.get_session("non_existent")
    assert result is None

def test_update_session(vector_store, mock_client):
    session_id = "test_update"
    existing_session = create_test_session(session_id)
    mock_point = Mock()
    mock_point.id = session_id
    mock_point.payload = existing_session
    mock_client.retrieve.return_value = [mock_point]
    mock_result = Mock()
    mock_result.status.name = "COMPLETED"
    mock_client.upsert.return_value = mock_result
    updates = {
        'duration_minutes': 45,
        'clinical_assessment': {
            'mood_rating': 8,
            'anxiety_level': 3
        }
    }
    result = vector_store.update_session(session_id, updates)
    assert result is True
    # Non-existent session
    mock_client.retrieve.return_value = []
    with pytest.raises(CBTVectorStoreError):
        vector_store.update_session("non_existent", updates)

def test_delete_session(vector_store, mock_client, collection_name):
    session_id = "test_delete"
    mock_result = Mock()
    mock_result.status.name = "COMPLETED"
    mock_client.delete.return_value = mock_result
    result = vector_store.delete_session(session_id)
    assert result is True
    mock_client.delete.assert_called_once()
    call_args = mock_client.delete.call_args
    assert call_args[1]["collection_name"] == collection_name
    assert session_id in call_args[1]["points_selector"]

def test_search_by_filters(vector_store, mock_client):
    mock_point = Mock()
    mock_point.id = "filtered_session"
    mock_point.payload = create_test_session("filtered_session")
    mock_client.scroll.return_value = ([mock_point], None)
    filters = {
        'data_source': 'local_system',
        'session_type': 'emotional_processing'
    }
    results = vector_store.search_by_filters(filters, limit=10)
    assert len(results) == 1
    assert results[0]["session_id"] == "filtered_session"
    # Empty filters
    with pytest.raises(CBTVectorStoreError):
        vector_store.search_by_filters({})

def test_collection_stats(vector_store, mock_client):
    mock_info = Mock()
    mock_info.status = Mock()
    mock_info.status.__str__ = Mock(return_value="green")
    mock_info.points_count = 150
    mock_info.vectors_count = 150
    mock_info.config.params.vectors.size = 3072
    mock_info.config.params.vectors.distance = Distance.COSINE
    mock_client.get_collection.return_value = mock_info
    stats = vector_store.get_collection_stats()
    assert stats["points_count"] == 150
    assert stats["schema_version"] == "3.3.0"

def test_health_check(vector_store, mock_client, collection_name):
    mock_collections = Mock()
    mock_collections.collections = [Mock(name=collection_name)]
    mock_client.get_collections.return_value = mock_collections
    mock_info = Mock()
    mock_info.status = "green"
    mock_info.points_count = 100
    mock_info.vectors_count = 100
    mock_client.get_collection.return_value = mock_info
    mock_client.scroll.return_value = ([], None)
    health = vector_store.health_check()
    assert "status" in health
    assert "checks" in health
    assert "timestamp" in health

def test_factory_function():
    with patch('cbt_journal.rag.vector_store.QdrantClient') as mock_qdrant_class:
        mock_client_instance = Mock()
        mock_qdrant_class.return_value = mock_client_instance
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client_instance.get_collections.return_value = mock_collections
        mock_info = Mock()
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = Mock()
        mock_info.config.params.vectors.size = 3072
        mock_client_instance.get_collection.return_value = mock_info
        vector_store = create_vector_store(
            qdrant_host="test_host",
            qdrant_port=9999,
            collection_name="test_collection"
        )
        assert isinstance(vector_store, CBTVectorStore)
        assert vector_store.collection_name == "test_collection"
        mock_qdrant_class.assert_called_once_with(host="test_host", port=9999)


def test_error_handling_connection_error(mock_client, cost_manager):
    # Qdrant connection error
    mock_client.get_collections.side_effect = Exception("Connection failed")
    with pytest.raises(CBTVectorStoreError):
        CBTVectorStore(
            qdrant_client=mock_client,
            cost_manager=cost_manager,
            collection_name="test"
        )

def test_error_handling_upsert_failure(collection_name, cost_manager):
    # Nuovo mock_client per isolamento
    client = Mock(spec=QdrantClient)
    # Collection exists
    mock_collection = Mock()
    mock_collection.name = collection_name
    mock_collections = Mock()
    mock_collections.collections = [mock_collection]
    client.get_collections.return_value = mock_collections
    # Collection info
    mock_info = Mock()
    mock_info.config = Mock()
    mock_info.config.params = Mock()
    mock_info.config.params.vectors = Mock()
    mock_info.config.params.vectors.size = 3072
    client.get_collection.return_value = mock_info
    # Upsert failure
    mock_result = Mock()
    mock_result.status.name = "FAILED"
    client.upsert.return_value = mock_result
    vector_store = CBTVectorStore(
        qdrant_client=client,
        cost_manager=cost_manager,
        collection_name=collection_name
    )
    with pytest.raises(CBTVectorStoreError):
        session_data = create_test_session()
        embedding = create_test_embedding()
        vector_store.store_session(session_data, embedding)

def test_search_with_score_threshold(vector_store, mock_client):
    # Test search with score threshold
    mock_point = Mock()
    mock_point.id = "high_score_session"
    mock_point.score = 0.9
    mock_point.payload = create_test_session("high_score_session")
    mock_results = Mock()
    mock_results.points = [mock_point]
    mock_client.query_points.return_value = mock_results
    
    query_embedding = create_test_embedding()
    results = vector_store.search_similar(query_embedding, score_threshold=0.8, limit=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.9
    
    # Verify score threshold was passed to client
    mock_client.query_points.assert_called_once()
    call_args = mock_client.query_points.call_args
    assert call_args[1]["score_threshold"] == 0.8

def test_search_with_complex_range_filters(vector_store, mock_client):
    # Test search with range filters
    mock_point = Mock()
    mock_point.id = "range_filtered_session"
    mock_point.score = 0.85
    mock_point.payload = create_test_session("range_filtered_session")
    mock_results = Mock()
    mock_results.points = [mock_point]
    mock_client.query_points.return_value = mock_results
    
    query_embedding = create_test_embedding()
    filters = {
        'duration_minutes': {'range': {'gte': 10, 'lte': 60}},
        'clinical_assessment.mood_rating': {'range': {'gte': 5}}
    }
    results = vector_store.search_similar(query_embedding, filters=filters, limit=10)
    assert len(results) == 1
    assert results[0]["session_id"] == "range_filtered_session"

def test_update_session_with_new_embedding(vector_store, mock_client):
    # Test updating session with new embedding
    session_id = "test_update_embedding"
    existing_session = create_test_session(session_id)
    mock_point = Mock()
    mock_point.id = session_id
    mock_point.payload = existing_session
    mock_client.retrieve.return_value = [mock_point]
    
    mock_result = Mock()
    mock_result.status.name = "COMPLETED"
    mock_client.upsert.return_value = mock_result
    
    updates = {'duration_minutes': 45}
    new_embedding = create_test_embedding()
    
    result = vector_store.update_session(session_id, updates, new_embedding)
    assert result is True
    
    # Verify upsert was called with new embedding
    mock_client.upsert.assert_called_once()
    call_args = mock_client.upsert.call_args
    upserted_point = call_args[1]["points"][0]
    assert upserted_point.vector == new_embedding

def test_search_by_filters_with_different_types(vector_store, mock_client):
    # Test search by filters with different data types
    mock_point = Mock()
    mock_point.id = "mixed_filters_session"
    mock_point.payload = create_test_session("mixed_filters_session")
    mock_client.scroll.return_value = ([mock_point], None)
    
    filters = {
        'data_source': 'local_system',  # string
        'duration_minutes': 30,         # int
        'clinical_assessment.mood_rating': {'range': {'gte': 5, 'lte': 8}}  # range
    }
    
    results = vector_store.search_by_filters(filters, limit=10)
    assert len(results) == 1
    assert results[0]["session_id"] == "mixed_filters_session"

def test_collection_stats_with_config(vector_store, mock_client):
    # Test collection stats including config details
    mock_info = Mock()
    mock_info.status = "green"
    mock_info.points_count = 100
    mock_info.vectors_count = 100
    mock_info.config = Mock()
    mock_info.config.params = Mock()
    mock_info.config.params.vectors = Mock()
    mock_info.config.params.vectors.size = 3072
    mock_info.config.params.vectors.distance = Distance.COSINE
    mock_client.get_collection.return_value = mock_info
    
    stats = vector_store.get_collection_stats()
    assert stats["points_count"] == 100
    assert stats["schema_version"] == "3.3.0"
    assert "config" in stats
    assert stats["config"]["vector_size"] == 3072
    assert "Cosine" in stats["config"]["distance"]

def test_health_check_complete_success(vector_store, mock_client, collection_name):
    # Test health check with all checks successful
    mock_collections = Mock()
    mock_collection_obj = Mock()
    mock_collection_obj.name = collection_name
    mock_collections.collections = [mock_collection_obj]
    mock_client.get_collections.return_value = mock_collections
    
    mock_info = Mock()
    mock_info.status = "green"
    mock_info.points_count = 100
    mock_info.vectors_count = 100
    mock_info.config = Mock()
    mock_info.config.params = Mock()
    mock_info.config.params.vectors = Mock()
    mock_info.config.params.vectors.size = 3072
    mock_info.config.params.vectors.distance = "cosine"
    mock_client.get_collection.return_value = mock_info
    
    # Mock successful scroll for index test
    mock_client.scroll.return_value = ([Mock()], None)
    
    health = vector_store.health_check()
    assert health["status"] == "healthy"
    assert health["checks"]["collection_access"] == "ok"
    assert health["checks"]["collection_exists"] == "ok"
    assert health["checks"]["basic_operations"] == "ok"
    assert health["checks"]["indexes"] == "ok"

def test_store_session_with_custom_session_id(vector_store, mock_client):
    # Test storing session with custom session ID
    custom_session_id = "custom_session_123"
    session_data = create_test_session()
    embedding = create_test_embedding()
    
    mock_result = Mock()
    mock_result.status.name = "COMPLETED"
    mock_client.upsert.return_value = mock_result
    
    result_session_id = vector_store.store_session(session_data, embedding, custom_session_id)
    assert result_session_id == custom_session_id
    
    # Verify the point was created with custom ID
    mock_client.upsert.assert_called_once()
    call_args = mock_client.upsert.call_args
    upserted_point = call_args[1]["points"][0]
    assert upserted_point.id == custom_session_id

def test_search_similar_empty_results(vector_store, mock_client):
    # Test search with no results
    mock_results = Mock()
    mock_results.points = []
    mock_client.query_points.return_value = mock_results
    
    query_embedding = create_test_embedding()
    results = vector_store.search_similar(query_embedding, limit=5)
    assert len(results) == 0
    assert isinstance(results, list)

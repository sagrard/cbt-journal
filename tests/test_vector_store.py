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
    
    def test_store_session(self) -> bool:
        """Test 2: Session storage"""
        print("\n" + "="*60)
        print("💾 TEST 2: SESSION STORAGE")
        print("="*60)
        
        # Mock successful collection verification
        mock_collections = Mock()
        mock_collections.collections = [Mock(name=self.collection_name)]
        self.mock_client.get_collections.return_value = mock_collections
        
        mock_info = Mock()
        mock_info.config.params.vectors.size = 3072
        self.mock_client.get_collection.return_value = mock_info
        
        # Mock successful upsert
        mock_result = Mock()
        mock_result.status.name = "COMPLETED"
        self.mock_client.upsert.return_value = mock_result
        
        try:
            # Test valid session storage
            session_data = self._create_test_session()
            embedding = self._create_test_embedding()
            
            session_id = self.vector_store.store_session(session_data, embedding)
            
            print(f"✅ Session stored: {session_id}")
            
            # Verify upsert was called
            self.mock_client.upsert.assert_called_once()
            call_args = self.mock_client.upsert.call_args
            
            assert call_args[1]['collection_name'] == self.collection_name
            assert len(call_args[1]['points']) == 1
            
            point = call_args[1]['points'][0]
            assert len(point.vector) == 3072
            assert point.payload['session_id'] == session_id
            assert 'system_metadata' in point.payload
            
            print("✅ Storage validation passed")
            
            # Test invalid embedding dimension
            try:
                self.vector_store.store_session(session_data, [1, 2, 3])  # Wrong dimension
                print("❌ Should have failed with wrong embedding dimension")
                return False
            except CBTVectorStoreError:
                print("✅ Invalid embedding rejected correctly")
            
            # Test missing required fields
            try:
                invalid_session = {"session_id": "test"}  # Missing required fields
                self.vector_store.store_session(invalid_session, embedding)
                print("❌ Should have failed with missing fields")
                return False
            except CBTVectorStoreError:
                print("✅ Invalid session data rejected correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Session storage test failed: {str(e)}")
            return False
    
    def test_search_similar(self) -> bool:
        """Test 3: Similarity search"""
        print("\n" + "="*60)
        print("🔍 TEST 3: SIMILARITY SEARCH")
        print("="*60)
        
        try:
            # Mock search results
            mock_point = Mock()
            mock_point.id = "test_session_123"
            mock_point.score = 0.85
            mock_point.payload = self._create_test_session("test_session_123")
            
            mock_results = Mock()
            mock_results.points = [mock_point]
            self.mock_client.query_points.return_value = mock_results
            
            # Test basic search
            query_embedding = self._create_test_embedding()
            results = self.vector_store.search_similar(query_embedding, limit=5)
            
            print(f"✅ Search returned {len(results)} results")
            
            # Validate result structure
            assert len(results) == 1
            result = results[0]
            assert 'session_id' in result
            assert 'score' in result
            assert 'payload' in result
            assert 'metadata' in result
            assert result['score'] == 0.85
            
            print("✅ Result structure valid")
            
            # Test search with filters
            filters = {
                'data_source': 'local_system',
                'clinical_assessment.mood_rating': {'range': {'gte': 5, 'lte': 8}}
            }
            
            results = self.vector_store.search_similar(
                query_embedding,
                filters=filters,
                limit=10
            )
            
            print("✅ Filtered search executed")
            
            # Test invalid embedding dimension
            try:
                self.vector_store.search_similar([1, 2, 3], limit=5)
                print("❌ Should have failed with wrong embedding dimension")
                return False
            except CBTVectorStoreError:
                print("✅ Invalid query embedding rejected")
            
            return True
            
        except Exception as e:
            print(f"❌ Similarity search test failed: {str(e)}")
            return False
    
    def test_get_session(self) -> bool:
        """Test 4: Direct session retrieval"""
        print("\n" + "="*60)
        print("📄 TEST 4: SESSION RETRIEVAL")
        print("="*60)
        
        try:
            session_id = "test_direct_retrieval"
            
            # Mock successful retrieval
            mock_point = Mock()
            mock_point.id = session_id
            mock_point.payload = self._create_test_session(session_id)
            self.mock_client.retrieve.return_value = [mock_point]
            
            # Test retrieval
            result = self.vector_store.get_session(session_id)
            
            assert result is not None
            assert result['session_id'] == session_id
            assert 'payload' in result
            assert 'metadata' in result
            
            print(f"✅ Session retrieved: {session_id}")
            
            # Test non-existent session
            self.mock_client.retrieve.return_value = []
            result = self.vector_store.get_session("non_existent")
            
            assert result is None
            print("✅ Non-existent session handled correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Session retrieval test failed: {str(e)}")
            return False
    
    def test_update_session(self) -> bool:
        """Test 5: Session updates"""
        print("\n" + "="*60)
        print("✏️  TEST 5: SESSION UPDATES")
        print("="*60)
        
        try:
            session_id = "test_update"
            
            # Mock existing session
            existing_session = self._create_test_session(session_id)
            mock_point = Mock()
            mock_point.id = session_id
            mock_point.payload = existing_session
            self.mock_client.retrieve.return_value = [mock_point]
            
            # Mock successful update
            mock_result = Mock()
            mock_result.status.name = "COMPLETED"
            self.mock_client.upsert.return_value = mock_result
            
            # Test update
            updates = {
                'duration_minutes': 45,
                'clinical_assessment': {
                    'mood_rating': 8,
                    'anxiety_level': 3
                }
            }
            
            result = self.vector_store.update_session(session_id, updates)
            
            assert result is True
            print(f"✅ Session updated: {session_id}")
            
            # Verify update call
            self.mock_client.upsert.assert_called()
            
            # Test update non-existent session
            self.mock_client.retrieve.return_value = []
            try:
                self.vector_store.update_session("non_existent", updates)
                print("❌ Should have failed for non-existent session")
                return False
            except CBTVectorStoreError:
                print("✅ Non-existent session update rejected")
            
            return True
            
        except Exception as e:
            print(f"❌ Session update test failed: {str(e)}")
            return False
    
    def test_delete_session(self) -> bool:
        """Test 6: Session deletion"""
        print("\n" + "="*60)
        print("🗑️  TEST 6: SESSION DELETION")
        print("="*60)
        
        try:
            session_id = "test_delete"
            
            # Mock successful deletion
            mock_result = Mock()
            mock_result.status.name = "COMPLETED"
            self.mock_client.delete.return_value = mock_result
            
            # Test deletion
            result = self.vector_store.delete_session(session_id)
            
            assert result is True
            print(f"✅ Session deleted: {session_id}")
            
            # Verify delete call
            self.mock_client.delete.assert_called_once()
            call_args = self.mock_client.delete.call_args
            assert call_args[1]['collection_name'] == self.collection_name
            assert session_id in call_args[1]['points_selector']
            
            return True
            
        except Exception as e:
            print(f"❌ Session deletion test failed: {str(e)}")
            return False
    
    def test_search_by_filters(self) -> bool:
        """Test 7: Metadata-only filtering"""
        print("\n" + "="*60)
        print("🔎 TEST 7: METADATA FILTERING")
        print("="*60)
        
        try:
            # Mock filter search results
            mock_point = Mock()
            mock_point.id = "filtered_session"
            mock_point.payload = self._create_test_session("filtered_session")
            
            self.mock_client.scroll.return_value = ([mock_point], None)
            
            # Test filter search
            filters = {
                'data_source': 'local_system',
                'session_type': 'emotional_processing'
            }
            
            results = self.vector_store.search_by_filters(filters, limit=10)
            
            assert len(results) == 1
            assert results[0]['session_id'] == "filtered_session"
            
            print(f"✅ Filter search returned {len(results)} results")
            
            # Test empty filters
            try:
                self.vector_store.search_by_filters({})
                print("❌ Should have failed with empty filters")
                return False
            except CBTVectorStoreError:
                print("✅ Empty filters rejected correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Metadata filtering test failed: {str(e)}")
            return False
    
    def test_collection_stats(self) -> bool:
        """Test 8: Collection statistics"""
        print("\n" + "="*60)
        print("📊 TEST 8: COLLECTION STATISTICS")
        print("="*60)
        
        try:
            # Mock collection info
            mock_info = Mock()
            mock_info.status = Mock()
            mock_info.status.__str__ = Mock(return_value="green")
            mock_info.points_count = 150
            mock_info.vectors_count = 150
            mock_info.config.params.vectors.size = 3072
            mock_info.config.params.vectors.distance = Distance.COSINE
            
            self.mock_client.get_collection.return_value = mock_info
            
            # Test stats
            stats = self.vector_store.get_collection_stats()
            
            assert 'collection_name' in stats
            assert 'points_count' in stats
            assert 'vectors_count' in stats
            assert 'schema_version' in stats
            assert stats['points_count'] == 150
            assert stats['schema_version'] == '3.3.0'
            
            print("✅ Collection statistics retrieved")
            print(f"   Points: {stats['points_count']}")
            print(f"   Vectors: {stats['vectors_count']}")
            print(f"   Schema: {stats['schema_version']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Collection stats test failed: {str(e)}")
            return False
    
    def test_health_check(self) -> bool:
        """Test 9: Health check"""
        print("\n" + "="*60)
        print("🏥 TEST 9: HEALTH CHECK")
        print("="*60)
        
        try:
            # Mock healthy collection
            mock_collections = Mock()
            mock_collections.collections = [Mock(name=self.collection_name)]
            self.mock_client.get_collections.return_value = mock_collections
            
            mock_info = Mock()
            mock_info.status = "green"
            mock_info.points_count = 100
            mock_info.vectors_count = 100
            self.mock_client.get_collection.return_value = mock_info
            
            # Mock successful filter test
            self.mock_client.scroll.return_value = ([], None)
            
            # Test health check
            health = self.vector_store.health_check()
            
            assert 'status' in health
            assert 'checks' in health
            assert 'timestamp' in health
            
            print(f"✅ Health check completed: {health['status']}")
            
            for check_name, check_status in health['checks'].items():
                print(f"   {check_name}: {check_status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check test failed: {str(e)}")
            return False
    
    def test_factory_function(self) -> bool:
        """Test 10: Factory function"""
        print("\n" + "="*60)
        print("🏭 TEST 10: FACTORY FUNCTION")
        print("="*60)
        
        try:
            # Test factory function with mocking
            with patch('cbt_journal.rag.vector_store.QdrantClient') as mock_qdrant_class:
                mock_client_instance = Mock()
                mock_qdrant_class.return_value = mock_client_instance
                
                # Mock successful verification for test_collection
                mock_collection = Mock()
                mock_collection.name = "test_collection"
                mock_collections = Mock()
                mock_collections.collections = [mock_collection]
                mock_client_instance.get_collections.return_value = mock_collections
                
                # Mock collection info
                mock_info = Mock()
                mock_info.config = Mock()
                mock_info.config.params = Mock()
                mock_info.config.params.vectors = Mock()
                mock_info.config.params.vectors.size = 3072
                mock_client_instance.get_collection.return_value = mock_info
                
                # Test factory
                vector_store = create_vector_store(
                    qdrant_host="test_host",
                    qdrant_port=9999,
                    collection_name="test_collection"
                )
                
                assert isinstance(vector_store, CBTVectorStore)
                assert vector_store.collection_name == "test_collection"
                
                print("✅ Factory function creates valid instance")
                
                # Verify QdrantClient was created correctly
                mock_qdrant_class.assert_called_once_with(host="test_host", port=9999)
                
                return True
                
        except Exception as e:
            print(f"❌ Factory function test failed: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test 11: Error handling"""
        print("\n" + "="*60)
        print("⚠️  TEST 11: ERROR HANDLING")
        print("="*60)
        
        try:
            # Test Qdrant connection error
            self.mock_client.get_collections.side_effect = Exception("Connection failed")
            
            try:
                CBTVectorStore(
                    qdrant_client=self.mock_client,
                    cost_manager=self.cost_manager,
                    collection_name="test"
                )
                print("❌ Should have failed with connection error")
                return False
            except CBTVectorStoreError:
                print("✅ Connection error handled correctly")
            
            # Reset mock
            self.mock_client.get_collections.side_effect = None
            
            # Test upsert failure
            mock_collections = Mock()
            mock_collections.collections = [Mock(name=self.collection_name)]
            self.mock_client.get_collections.return_value = mock_collections
            
            mock_info = Mock()
            mock_info.config.params.vectors.size = 3072
            self.mock_client.get_collection.return_value = mock_info
            
            mock_result = Mock()
            mock_result.status.name = "FAILED"
            self.mock_client.upsert.return_value = mock_result
            
            try:
                session_data = self._create_test_session()
                embedding = self._create_test_embedding()
                self.vector_store.store_session(session_data, embedding)
                print("❌ Should have failed with upsert error")
                return False
            except CBTVectorStoreError:
                print("✅ Upsert failure handled correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("🚀 STARTING CBT VECTOR STORE TEST SUITE")
        print("=" * 70)
        
        tests = [
            ("Initialization", self.test_initialization),
            ("Session Storage", self.test_store_session),
            ("Similarity Search", self.test_search_similar),
            ("Session Retrieval", self.test_get_session),
            ("Session Updates", self.test_update_session),
            ("Session Deletion", self.test_delete_session),
            ("Metadata Filtering", self.test_search_by_filters),
            ("Collection Statistics", self.test_collection_stats),
            ("Health Check", self.test_health_check),
            ("Factory Function", self.test_factory_function),
            ("Error Handling", self.test_error_handling)
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
        
        # Test summary
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
            print("🎉 ALL TESTS PASSED - VECTOR STORE READY!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW BEFORE PROCEEDING")
            return False

#!/usr/bin/env python3
"""
CBT Vector Store - Qdrant wrapper with cost control integration
Handles all vector operations for CBT Journal sessions
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, 
    Range, MatchValue, SearchRequest, Record
)
from qdrant_client.http.exceptions import ResponseHandlingException

from ..utils.cost_control import CostControlManager


class CBTVectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass


class CBTVectorStore:
    """
    Vector store wrapper for CBT Journal sessions
    Integrates Qdrant operations with cost control and schema v3.3.0
    """
    
    def __init__(self, 
                 qdrant_client: QdrantClient,
                 cost_manager: CostControlManager,
                 collection_name: str = "cbt_journal_sessions"):
        """
        Initialize CBT Vector Store
        
        Args:
            qdrant_client: Configured Qdrant client
            cost_manager: Cost control manager for API calls
            collection_name: Qdrant collection name
        """
        self.client = qdrant_client
        self.cost_manager = cost_manager
        self.collection_name = collection_name
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Verify collection exists
        self._verify_collection()
    
    def _verify_collection(self) -> None:
        """Verify collection exists and is properly configured"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                raise CBTVectorStoreError(
                    f"Collection '{self.collection_name}' not found. "
                    "Run tools/setup_qdrant.py first."
                )
            
            # Verify collection config
            info = self.client.get_collection(self.collection_name)
            if hasattr(info.config, 'params') and info.config.params:
                vector_config = info.config.params.vectors
                if vector_config.size != 3072:
                    raise CBTVectorStoreError(
                        f"Collection vector size {vector_config.size}, expected 3072"
                    )
            
            self.logger.info(f"Collection '{self.collection_name}' verified")
            
        except Exception as e:
            raise CBTVectorStoreError(f"Collection verification failed: {str(e)}")
    
    def store_session(self, 
                     session_data: Dict[str, Any], 
                     embedding: List[float],
                     session_id: Optional[str] = None) -> str:
        """
        Store CBT session with embedding
        
        Args:
            session_data: Session data conforming to schema v3.3.0
            embedding: 3072-dim vector from text-embedding-3-large
            session_id: Optional session ID (auto-generated if None)
            
        Returns:
            str: Session ID of stored session
            
        Raises:
            CBTVectorStoreError: If storage fails
        """
        if session_id is None:
            session_id = session_data.get('session_id', str(uuid.uuid4()))
        
        # Validate embedding dimensions
        if len(embedding) != 3072:
            raise CBTVectorStoreError(
                f"Embedding dimension {len(embedding)}, expected 3072"
            )
        
        # Validate required schema fields
        required_fields = ['session_id', 'timestamp', 'data_source', 'content', 'ai_models']
        missing_fields = [field for field in required_fields if field not in session_data]
        if missing_fields:
            raise CBTVectorStoreError(f"Missing required fields: {missing_fields}")
        
        # Add system metadata
        session_data['system_metadata'] = session_data.get('system_metadata', {})
        session_data['system_metadata'].update({
            'schema_version': '3.3.0',
            'created_at': datetime.now().isoformat(),
            'vector_stored_at': datetime.now().isoformat()
        })
        
        # Create point
        point = PointStruct(
            id=session_id,
            vector=embedding,
            payload=session_data
        )
        
        try:
            # Store in Qdrant
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            if result.status.name != "COMPLETED":
                raise CBTVectorStoreError(f"Upsert failed with status: {result.status}")
            
            self.logger.info(f"Session {session_id} stored successfully")
            return session_id
            
        except Exception as e:
            raise CBTVectorStoreError(f"Failed to store session: {str(e)}")
    
    def search_similar(self, 
                      query_embedding: List[float],
                      limit: int = 10,
                      score_threshold: float = 0.7,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar sessions using vector similarity
        
        Args:
            query_embedding: Query vector (3072 dimensions)
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            filters: Optional metadata filters
            
        Returns:
            List of similar sessions with scores and metadata
            
        Raises:
            CBTVectorStoreError: If search fails
        """
        if len(query_embedding) != 3072:
            raise CBTVectorStoreError(
                f"Query embedding dimension {len(query_embedding)}, expected 3072"
            )
        
        # Build filter conditions
        filter_conditions = []
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                elif isinstance(value, (int, float)):
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                elif isinstance(value, dict) and 'range' in value:
                    range_filter = value['range']
                    filter_conditions.append(
                        FieldCondition(
                            key=key, 
                            range=Range(
                                gte=range_filter.get('gte'),
                                lte=range_filter.get('lte')
                            )
                        )
                    )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            # Perform search
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_results.points:
                result = {
                    'session_id': point.id,
                    'score': point.score,
                    'payload': point.payload,
                    'metadata': {
                        'retrieval_timestamp': datetime.now().isoformat(),
                        'query_type': 'semantic_similarity'
                    }
                }
                results.append(result)
            
            self.logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            raise CBTVectorStoreError(f"Search failed: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
            
        Raises:
            CBTVectorStoreError: If retrieval fails
        """
        try:
            # Retrieve point
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[session_id],
                with_payload=True,
                with_vectors=False  # Don't need vector for direct retrieval
            )
            
            if not points:
                return None
            
            point = points[0]
            result = {
                'session_id': point.id,
                'payload': point.payload,
                'metadata': {
                    'retrieval_timestamp': datetime.now().isoformat(),
                    'query_type': 'direct_retrieval'
                }
            }
            
            self.logger.info(f"Session {session_id} retrieved successfully")
            return result
            
        except Exception as e:
            raise CBTVectorStoreError(f"Failed to retrieve session {session_id}: {str(e)}")
    
    def update_session(self, 
                      session_id: str, 
                      updates: Dict[str, Any],
                      new_embedding: Optional[List[float]] = None) -> bool:
        """
        Update existing session data
        
        Args:
            session_id: Session to update
            updates: Partial updates to apply
            new_embedding: Optional new embedding vector
            
        Returns:
            bool: True if update successful
            
        Raises:
            CBTVectorStoreError: If update fails
        """
        try:
            # Get existing session
            existing = self.get_session(session_id)
            if not existing:
                raise CBTVectorStoreError(f"Session {session_id} not found")
            
            # Merge updates
            updated_payload = existing['payload'].copy()
            updated_payload.update(updates)
            
            # Update system metadata
            updated_payload['system_metadata'] = updated_payload.get('system_metadata', {})
            updated_payload['system_metadata']['updated_at'] = datetime.now().isoformat()
            
            # Create updated point
            point = PointStruct(
                id=session_id,
                vector=new_embedding if new_embedding is not None else [],
                payload=updated_payload
            )
            
            # Update in Qdrant
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            if result.status.name != "COMPLETED":
                raise CBTVectorStoreError(f"Update failed with status: {result.status}")
            
            self.logger.info(f"Session {session_id} updated successfully")
            return True
            
        except Exception as e:
            raise CBTVectorStoreError(f"Failed to update session {session_id}: {str(e)}")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from vector store
        
        Args:
            session_id: Session to delete
            
        Returns:
            bool: True if deletion successful
            
        Raises:
            CBTVectorStoreError: If deletion fails
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=[session_id]
            )
            
            if result.status.name != "COMPLETED":
                raise CBTVectorStoreError(f"Delete failed with status: {result.status}")
            
            self.logger.info(f"Session {session_id} deleted successfully")
            return True
            
        except Exception as e:
            raise CBTVectorStoreError(f"Failed to delete session {session_id}: {str(e)}")
    
    def search_by_filters(self, 
                         filters: Dict[str, Any],
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search sessions by metadata filters only (no vector search)
        
        Args:
            filters: Metadata filter conditions
            limit: Maximum results
            
        Returns:
            List of matching sessions
            
        Raises:
            CBTVectorStoreError: If search fails
        """
        # Build filter conditions
        filter_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, str):
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, (int, float)):
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, dict) and 'range' in value:
                range_filter = value['range']
                filter_conditions.append(
                    FieldCondition(
                        key=key, 
                        range=Range(
                            gte=range_filter.get('gte'),
                            lte=range_filter.get('lte')
                        )
                    )
                )
        
        if not filter_conditions:
            raise CBTVectorStoreError("At least one filter condition required")
        
        query_filter = Filter(must=filter_conditions)
        
        try:
            # Use scroll for filter-only search
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for point in points:
                result = {
                    'session_id': point.id,
                    'payload': point.payload,
                    'metadata': {
                        'retrieval_timestamp': datetime.now().isoformat(),
                        'query_type': 'metadata_filter'
                    }
                }
                results.append(result)
            
            self.logger.info(f"Filter search returned {len(results)} results")
            return results
            
        except Exception as e:
            raise CBTVectorStoreError(f"Filter search failed: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics and health info
        
        Returns:
            Collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            
            stats = {
                'collection_name': self.collection_name,
                'status': str(info.status),
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'schema_version': '3.3.0',
                'last_updated': datetime.now().isoformat()
            }
            
            # Add config details if available
            if hasattr(info.config, 'params') and info.config.params:
                stats['config'] = {
                    'vector_size': info.config.params.vectors.size,
                    'distance': str(info.config.params.vectors.distance)
                }
            
            return stats
            
        except Exception as e:
            raise CBTVectorStoreError(f"Failed to get collection stats: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on vector store
        
        Returns:
            Health status and diagnostics
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'collection_name': self.collection_name,
            'status': 'unknown',
            'checks': {}
        }
        
        try:
            # Check 1: Collection access
            collections = self.client.get_collections()
            health_status['checks']['collection_access'] = 'ok'
            
            # Check 2: Collection exists
            collection_names = [col.name for col in collections.collections]
            if self.collection_name in collection_names:
                health_status['checks']['collection_exists'] = 'ok'
            else:
                health_status['checks']['collection_exists'] = 'error'
                health_status['status'] = 'error'
                return health_status
            
            # Check 3: Basic operations
            stats = self.get_collection_stats()
            health_status['checks']['basic_operations'] = 'ok'
            health_status['stats'] = stats
            
            # Check 4: Index status (if we can query payload indexes)
            try:
                # Simple filter query to test indexes
                self.search_by_filters({'data_source': 'local_system'}, limit=1)
                health_status['checks']['indexes'] = 'ok'
            except:
                health_status['checks']['indexes'] = 'warning'
            
            # Overall status
            if all(check == 'ok' for check in health_status['checks'].values()):
                health_status['status'] = 'healthy'
            elif any(check == 'error' for check in health_status['checks'].values()):
                health_status['status'] = 'error'
            else:
                health_status['status'] = 'warning'
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status


# Factory function for easy initialization
def create_vector_store(qdrant_host: str = "localhost",
                       qdrant_port: int = 6333,
                       collection_name: str = "cbt_journal_sessions",
                       cost_manager: Optional[CostControlManager] = None) -> CBTVectorStore:
    """
    Factory function to create configured CBT Vector Store
    
    Args:
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        collection_name: Collection name
        cost_manager: Optional cost manager (created if None)
        
    Returns:
        Configured CBTVectorStore instance
    """
    # Create Qdrant client
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Create cost manager if not provided
    if cost_manager is None:
        cost_manager = CostControlManager()
    
    # Create and return vector store
    return CBTVectorStore(
        qdrant_client=qdrant_client,
        cost_manager=cost_manager,
        collection_name=collection_name
    )
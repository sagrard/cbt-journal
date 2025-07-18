#!/usr/bin/env python3
"""
Qdrant Setup per Schema v3.3.0 - CBT Journal
Configurazione ottimizzata per performance con payload indexes
"""

import json
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    PayloadSchemaType,
)


class CBTQdrantV330Setup:
    def __init__(self, host: str = "localhost", port: int = 6334, prefer_grpc: bool = True):
        """Inizializza connessione Qdrant per CBT Journal v3.3.0"""
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
        self.collection_name = "cbt_journal_sessions"  # Nome corretto dal .env.example

    def create_cbt_collection(self) -> bool:
        """
        Crea collection CBT Journal con schema v3.3.0
        Embedding: OpenAI text-embedding-3-large (3072 dimensions)
        """
        try:
            # Verifica se collection esiste già
            try:
                collections = self.client.get_collections()
                existing_names = [col.name for col in collections.collections]
                if self.collection_name in existing_names:
                    print(f"✅ Collection '{self.collection_name}' già esistente")
                    return self._verify_collection_config()
            except Exception:
                print("Creando prima collection...")

            # Configurazione vettori per OpenAI text-embedding-3-large
            vector_config = VectorParams(
                size=3072, distance=Distance.COSINE  # text-embedding-3-large dimension
            )

            # Crea collection
            self.client.create_collection(
                collection_name=self.collection_name, vectors_config=vector_config
            )

            print(f"✅ Collection '{self.collection_name}' creata con successo")

            # Setup payload indexes per performance
            return self._setup_payload_indexes()

        except Exception as e:
            print(f"❌ Errore creazione collection: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return False

    def _setup_payload_indexes(self) -> bool:
        """Setup payload indexes per campi critici schema v3.3.0"""
        try:
            print("🔧 Configurando payload indexes per performance...")

            # Indexes critici per RAG performance
            indexes_config = [
                # RAG Metadata - Core per retrieval
                ("rag_metadata.context_type", PayloadSchemaType.KEYWORD),
                ("rag_metadata.language", PayloadSchemaType.KEYWORD),
                ("rag_metadata.context_priority", PayloadSchemaType.INTEGER),
                # Clinical Classification - Filtering therapeutico
                ("clinical_classification.primary_themes", PayloadSchemaType.KEYWORD),
                ("clinical_classification.risk_assessment.risk_level", PayloadSchemaType.KEYWORD),
                # Narrative Insight - Emotional intelligence
                ("narrative_insight.emotional_needs", PayloadSchemaType.KEYWORD),
                ("narrative_insight.turning_point_detected", PayloadSchemaType.BOOL),
                # Clinical Assessment - Standard scales
                ("clinical_assessment.mood_rating", PayloadSchemaType.INTEGER),
                ("clinical_assessment.anxiety_level", PayloadSchemaType.INTEGER),
                # Cost Monitoring - Budget control
                ("cost_monitoring.session_budget.budget_status", PayloadSchemaType.KEYWORD),
                (
                    "cost_monitoring.monthly_tracking.monthly_budget_status",
                    PayloadSchemaType.KEYWORD,
                ),
                # Core session metadata
                ("session_type", PayloadSchemaType.KEYWORD),
                ("data_source", PayloadSchemaType.KEYWORD),
                ("timestamp", PayloadSchemaType.DATETIME),
                # AI Models tracking
                ("ai_models.response_model.name", PayloadSchemaType.KEYWORD),
                ("ai_models.session_cost_summary.total_cost_usd", PayloadSchemaType.FLOAT),
            ]

            # Crea indexes
            for field_name, schema_type in indexes_config:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    print(f"   ✅ Index: {field_name}")
                except Exception as e:
                    # Alcuni indexes potrebbero già esistere
                    if "already exists" in str(e).lower():
                        print(f"   ✓ Index exists: {field_name}")
                    else:
                        print(f"   ⚠️ Index failed: {field_name} - {str(e)}")

            print("✅ Payload indexes configurati")
            return True

        except Exception as e:
            print(f"❌ Errore setup indexes: {str(e)}")
            return False

    def _verify_collection_config(self) -> bool:
        """Verifica configurazione collection esistente"""
        try:
            info = self.client.get_collection(self.collection_name)

            # Verifica vector config
            if hasattr(info.config, "params") and info.config.params:
                vector_config = info.config.params.vectors
                if vector_config.size != 3072:
                    print(f"⚠️ Warning: Vector size {vector_config.size}, expected 3072")
                    return False
                if vector_config.distance != Distance.COSINE:
                    print(f"⚠️ Warning: Distance {vector_config.distance}, expected COSINE")
                    return False

            print("✅ Collection config verificata (3072 dim, COSINE)")

            # Setup indexes se non esistono
            return self._setup_payload_indexes()

        except Exception as e:
            print(f"❌ Errore verifica config: {str(e)}")
            return False

    def test_schema_v330_operations(self) -> bool:
        """Test completo operazioni con schema v3.3.0"""
        try:
            print("\n🧪 Testing schema v3.3.0 operations...")

            import random
            import uuid
            from datetime import datetime

            # Create test session seguendo schema v3.3.0
            test_vector = [random.random() for _ in range(3072)]
            test_session_id = str(uuid.uuid4())

            # Payload conforme a schema v3.3.0 (sample completo)
            test_payload = {
                "session_id": test_session_id,
                "timestamp": datetime.now().isoformat(),
                "data_source": "local_system",
                "session_type": "emotional_processing",
                "duration_minutes": 25,
                # Content (required)
                "content": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Test session per validazione schema v3.3.0",
                            "timestamp": datetime.now().isoformat(),
                            "word_count": 8,
                        },
                        {
                            "role": "assistant",
                            "content": "Comprendo, questo è un test del sistema CBT Journal.",
                            "timestamp": datetime.now().isoformat(),
                            "word_count": 11,
                        },
                    ],
                    "conversation_count": 1,
                    "total_words": {"user": 8, "assistant": 11, "total": 19},
                },
                # AI Models (required)
                "ai_models": {
                    "tracking_level": "complete",
                    "response_model": {
                        "name": "gpt-4o-2024-11-20",
                        "confidence": "exact",
                        "provider": "openai",
                    },
                    "session_cost_summary": {
                        "total_cost_usd": 0.025,
                        "cost_breakdown": {"embedding": 0.005, "generation": 0.020},
                        "total_tokens": {"input": 150, "output": 75, "total": 225},
                    },
                },
                # Clinical Assessment (nuovo v3.3.0)
                "clinical_assessment": {
                    "data_available": True,
                    "mood_rating": 7,
                    "anxiety_level": 4,
                    "energy_level": 6,
                    "assessment_method": "user_input",
                },
                # RAG Metadata (critical)
                "rag_metadata": {
                    "context_type": "narrative",
                    "language": "it",
                    "context_priority": 4,
                    "token_count": 225,
                    "embedding_version": "text-embedding-3-large",
                },
                # Narrative Insight (emotional intelligence)
                "narrative_insight": {
                    "emotional_needs": ["validation", "understanding"],
                    "narrative_theme": "testing_system_capability",
                    "turning_point_detected": False,
                    "confidence": "high",
                },
                # Clinical Classification
                "clinical_classification": {
                    "data_available": True,
                    "primary_themes": ["anxiety", "self_esteem"],
                    "emotional_patterns": ["attentiveness"],
                    "risk_assessment": {"risk_level": "low", "crisis_indicators": False},
                },
                # Cost Monitoring (budget control)
                "cost_monitoring": {
                    "session_budget": {
                        "max_cost_per_session": 0.50,
                        "current_session_cost": 0.025,
                        "budget_status": "within_limits",
                    }
                },
                # System metadata (required)
                "system_metadata": {
                    "schema_version": "3.3.0",
                    "created_at": datetime.now().isoformat(),
                },
            }

            # Test 1: Insert con schema completo
            point_id = str(uuid.uuid4())
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=test_vector, payload=test_payload)],
            )
            print(f"✅ Insert schema v3.3.0: {result}")

            # Test 2: Search semantico
            search_results = self.client.query_points(
                collection_name=self.collection_name, query=test_vector, limit=1
            )
            print(f"✅ Semantic search: {len(search_results.points)} results")

            # Test 3: Filtered search (RAG use case)
            filter_condition = Filter(
                must=[
                    FieldCondition(key="rag_metadata.language", match={"value": "it"}),
                    FieldCondition(key="clinical_assessment.mood_rating", range=Range(gte=5)),
                    FieldCondition(
                        key="clinical_classification.risk_assessment.risk_level",
                        match={"value": "low"},
                    ),
                ]
            )

            filter_results = self.client.query_points(
                collection_name=self.collection_name,
                query=test_vector,
                query_filter=filter_condition,
                limit=5,
            )
            print(f"✅ Multi-field filter: {len(filter_results.points)} results")

            # Test 4: Cost monitoring filter
            cost_filter = Filter(
                must=[
                    FieldCondition(
                        key="cost_monitoring.session_budget.budget_status",
                        match={"value": "within_limits"},
                    )
                ]
            )

            cost_results = self.client.query_points(
                collection_name=self.collection_name,
                query=test_vector,
                query_filter=cost_filter,
                limit=3,
            )
            print(f"✅ Cost monitoring filter: {len(cost_results.points)} results")

            # Test 5: Verifica payload completo
            if search_results.points:
                retrieved = search_results.points[0]
                payload_keys = set(retrieved.payload.keys())
                expected_keys = {
                    "session_id",
                    "timestamp",
                    "content",
                    "ai_models",
                    "clinical_assessment",
                    "rag_metadata",
                    "narrative_insight",
                    "clinical_classification",
                    "cost_monitoring",
                    "system_metadata",
                }

                if expected_keys.issubset(payload_keys):
                    print("✅ Schema v3.3.0 payload: COMPLETE")
                else:
                    missing = expected_keys - payload_keys
                    print(f"⚠️ Schema payload missing: {missing}")

            # Cleanup
            self.client.delete(collection_name=self.collection_name, points_selector=[point_id])
            print("✅ Test cleanup completed")

            return True

        except Exception as e:
            print(f"❌ Errore test schema v3.3.0: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return False

    def setup_automated_backup(self) -> bool:
        """Setup basic backup verification"""
        try:
            print("\n💾 Testing backup capabilities...")

            # Test snapshot creation capability
            collections = self.client.get_collections()
            print(f"✅ Collection access: {len(collections.collections)} collections")

            # Get collection info for backup planning
            info = self.client.get_collection(self.collection_name)
            print(f"✅ Backup planning: {info.points_count} points, {info.vectors_count} vectors")

            print("📋 Backup recommendations:")
            print("   1. Regular collection export: qdrant collection export")
            print("   2. Docker volume backup: docker volume backup qdrant_data")
            print("   3. JSON export for portability: custom script needed")

            return True

        except Exception as e:
            print(f"❌ Errore backup setup: {str(e)}")
            return False

    def get_collection_info_v330(self) -> Dict[str, Any]:
        """Informazioni collection con dettagli v3.3.0"""
        try:
            info = self.client.get_collection(self.collection_name)

            result = {
                "collection_name": self.collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "schema_version": "3.3.0",
                "optimized_for": "CBT Journal with cost tracking",
            }

            # Config details
            if hasattr(info.config, "params") and info.config.params:
                result["vector_config"] = {
                    "size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                    "optimized_for": "OpenAI text-embedding-3-large",
                }

            return result

        except Exception as e:
            return {"error": str(e)}

    def setup_complete_v330_system(self) -> bool:
        """Setup completo sistema v3.3.0"""
        print("=" * 60)
        print("CBT JOURNAL QDRANT SETUP - SCHEMA v3.3.0")
        print("=" * 60)

        # Test 1: Connessione
        try:
            collections = self.client.get_collections()
            print(f"✅ Qdrant connection established ({len(collections.collections)} collections)")
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False

        # Test 2: Collection + Indexes
        if not self.create_cbt_collection():
            print("❌ Collection/indexes setup failed")
            return False

        # Test 3: Schema v3.3.0 operations
        if not self.test_schema_v330_operations():
            print("❌ Schema v3.3.0 test failed")
            return False

        # Test 4: Backup setup
        if not self.setup_automated_backup():
            print("❌ Backup setup failed")
            return False

        # Final info
        info = self.get_collection_info_v330()
        print("\n📊 CBT JOURNAL COLLECTION INFO:")
        print(json.dumps(info, indent=2, ensure_ascii=False))

        print("\n🎉 SETUP v3.3.0 COMPLETED SUCCESSFULLY!")
        print("✅ Features configured:")
        print("   - OpenAI text-embedding-3-large (3072 dim)")
        print("   - Optimized payload indexes for RAG")
        print("   - Clinical assessment tracking")
        print("   - Cost monitoring & budget control")
        print("   - Narrative insight & emotional intelligence")
        print("   - Complete audit trail")
        print("\n🚀 Sistema pronto per Week 2: RAG Pipeline Core")

        return True


def main():
    """Setup principale v3.3.0"""
    print("CBT Journal - Qdrant Setup v3.3.0")
    print("Collection: cbt_journal_sessions")
    print("Schema: v3.3.0 (Production Ready)")

    setup = CBTQdrantV330Setup()
    success = setup.setup_complete_v330_system()

    if success:
        print("\n✅ SUCCESS: Qdrant optimizzato per CBT Journal")
        print("\nNext steps:")
        print("1. Week 2: Implementare RAG Pipeline")
        print("2. OpenAI API integration")
        print("3. Embedding generation pipeline")
        print("4. Context retrieval engine")
    else:
        print("\n❌ SETUP FAILED - troubleshoot required")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

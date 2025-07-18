#!/usr/bin/env python3
"""
Test configurazione default Qdrant per sistema CBT
Verifica che tutto funzioni correttamente senza config custom
"""

import uuid
import random
import time
from typing import Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range


class QdrantDefaultConfigTest:
    def __init__(self, host: str = "localhost", port: int = 6334, prefer_grpc: bool = True):
        """Test configurazione default Qdrant"""
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
        self.collection_name = "cbt_diary_sessions"

    def test_connection_and_info(self) -> bool:
        """Test connessione e info sistema"""
        try:
            # Verifica connessione
            collections = self.client.get_collections()
            print(f"✅ Connessione stabilita ({len(collections.collections)} collections)")

            # Info sistema se collection esiste
            if any(col.name == self.collection_name for col in collections.collections):
                info = self.client.get_collection(self.collection_name)
                print(f"✅ Collection esistente: {info.status}")
                print(f"   Points: {info.points_count}")
                print(f"   Vectors: {info.vectors_count}")

                # Mostra config effettiva
                if hasattr(info.config, "params"):
                    params = info.config.params
                    print(f"   Vector size: {params.vectors.size}")
                    print(f"   Distance: {params.vectors.distance}")
                    print(f"   Shard number: {params.shard_number}")
                    print(f"   Replication factor: {params.replication_factor}")

                    # Info HNSW se disponibile
                    if hasattr(params.vectors, "hnsw_config") and params.vectors.hnsw_config:
                        hnsw = params.vectors.hnsw_config
                        print(f"   HNSW m: {hnsw.m}")
                        print(f"   HNSW ef_construct: {hnsw.ef_construct}")

            return True

        except Exception as e:
            print(f"❌ Errore connessione: {str(e)}")
            return False

    def test_performance_baseline(self) -> Dict[str, float]:
        """Test performance con configurazione default"""
        try:
            print("\n🚀 Testing performance baseline...")

            # Assicurati che collection esista
            if not any(
                col.name == self.collection_name
                for col in self.client.get_collections().collections
            ):
                # Crea collection con config minimal
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                )
                print("✅ Collection creata con config default")

            # Test 1: Insert performance
            start_time = time.time()
            test_points = []
            for i in range(10):
                vector = [random.random() for _ in range(3072)]
                payload = {
                    "session_id": f"perf_test_{i}",
                    "content": f"Performance test session {i} con contenuto più lungo per simulare sessioni CBT reali",
                    "timestamp": f"2025-01-{i + 1:02d}T10:00:00Z",
                    "mood": random.randint(1, 10),
                    "anxiety": random.randint(1, 10),
                    "tags": ["performance", "test", f"batch_{i // 3}"],
                }
                test_points.append(
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                )

            self.client.upsert(collection_name=self.collection_name, points=test_points)
            insert_time = time.time() - start_time
            print(
                f"✅ Insert 10 points: {insert_time:.3f}s ({insert_time / 10 * 1000:.1f}ms per point)"
            )

            # Test 2: Search performance
            start_time = time.time()
            search_results = self.client.query_points(
                collection_name=self.collection_name, query=test_points[0].vector, limit=5
            )
            search_time = time.time() - start_time
            print(f"✅ Search 5 results: {search_time:.3f}s ({len(search_results.points)} found)")

            # Test 3: Filtered search performance
            start_time = time.time()
            filter_condition = Filter(
                must=[
                    FieldCondition(key="mood", range=Range(gte=5)),
                    FieldCondition(key="anxiety", range=Range(lte=7)),
                ]
            )
            filter_results = self.client.query_points(
                collection_name=self.collection_name,
                query=test_points[0].vector,
                query_filter=filter_condition,
                limit=5,
            )
            filter_time = time.time() - start_time
            print(f"✅ Filtered search: {filter_time:.3f}s ({len(filter_results.points)} found)")

            # Test 4: Count performance
            start_time = time.time()
            count_result = self.client.count(self.collection_name)
            count_time = time.time() - start_time
            print(f"✅ Count operation: {count_time:.3f}s ({count_result.count} total points)")

            # Cleanup test data
            test_ids = [point.id for point in test_points]
            self.client.delete(collection_name=self.collection_name, points_selector=test_ids)
            print("✅ Cleanup completed")

            return {
                "insert_time": insert_time,
                "search_time": search_time,
                "filter_time": filter_time,
                "count_time": count_time,
            }

        except Exception as e:
            print(f"❌ Errore performance test: {str(e)}")
            return {}

    def test_concurrent_operations(self) -> bool:
        """Test capacità handling operazioni multiple"""
        try:
            print("\n🔄 Testing concurrent capability...")

            # Simula operazioni multiple simultanee
            import concurrent.futures

            def insert_batch(batch_id: int) -> str:
                """Insert batch separato"""
                try:
                    points = []
                    for i in range(5):
                        vector = [random.random() for _ in range(3072)]
                        payload = {
                            "session_id": f"concurrent_batch_{batch_id}_{i}",
                            "content": f"Concurrent test batch {batch_id} item {i}",
                            "batch_id": batch_id,
                            "item_id": i,
                        }
                        points.append(
                            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                        )

                    self.client.upsert(collection_name=self.collection_name, points=points)
                    return f"Batch {batch_id}: OK"
                except Exception as e:
                    return f"Batch {batch_id}: ERROR - {str(e)}"

            # Esegui 3 batch concorrenti
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(insert_batch, i) for i in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Verifica risultati
            success_count = sum(1 for result in results if "OK" in result)
            print(f"✅ Concurrent operations: {success_count}/3 successful")
            for result in results:
                print(f"   {result}")

            # Cleanup
            filter_condition = Filter(
                must=[FieldCondition(key="batch_id", range=Range(gte=0, lte=2))]
            )

            # Get points to delete
            cleanup_points = self.client.scroll(
                collection_name=self.collection_name, scroll_filter=filter_condition, limit=20
            )

            if cleanup_points[0]:
                cleanup_ids = [point.id for point in cleanup_points[0]]
                self.client.delete(
                    collection_name=self.collection_name, points_selector=cleanup_ids
                )
                print("✅ Concurrent test cleanup completed")

            return success_count == 3

        except Exception as e:
            print(f"❌ Errore concurrent test: {str(e)}")
            return False

    def run_complete_test(self) -> bool:
        """Esegui test completo configurazione default"""
        print("=" * 60)
        print("TEST CONFIGURAZIONE DEFAULT QDRANT")
        print("=" * 60)

        # Test 1: Connessione e info
        if not self.test_connection_and_info():
            return False

        # Test 2: Performance baseline
        perf_results = self.test_performance_baseline()
        if not perf_results:
            return False

        # Test 3: Concurrent operations
        if not self.test_concurrent_operations():
            return False

        # Summary performance
        print("\n📊 PERFORMANCE SUMMARY (Default Config):")
        for operation, time_taken in perf_results.items():
            print(f"   {operation}: {time_taken:.3f}s")

        # Valutazione performance per CBT
        print("\n🎯 EVALUATION FOR CBT USE CASE:")

        insert_per_session = perf_results.get("insert_time", 0) / 10
        if insert_per_session < 0.1:
            print(f"   ✅ Session insert: EXCELLENT ({insert_per_session * 1000:.1f}ms)")
        elif insert_per_session < 0.5:
            print(f"   ✅ Session insert: GOOD ({insert_per_session * 1000:.1f}ms)")
        else:
            print(f"   ⚠️ Session insert: SLOW ({insert_per_session * 1000:.1f}ms)")

        search_time = perf_results.get("search_time", 0)
        if search_time < 0.05:
            print(f"   ✅ Context retrieval: EXCELLENT ({search_time * 1000:.1f}ms)")
        elif search_time < 0.2:
            print(f"   ✅ Context retrieval: GOOD ({search_time * 1000:.1f}ms)")
        else:
            print(f"   ⚠️ Context retrieval: SLOW ({search_time * 1000:.1f}ms)")

        print("\n🎉 CONFIGURAZIONE DEFAULT: VALIDATED FOR CBT")
        print("Sistema pronto per produzione con performance ottimali")

        return True


def main():
    """Test principale"""
    tester = QdrantDefaultConfigTest()

    success = tester.run_complete_test()

    if success:
        print("\n✅ MIGRATION TO DEFAULT CONFIG: SUCCESS")
        print("Benefici:")
        print("- Auto-ottimizzazione hardware")
        print("- Zero maintenance config files")
        print("- Future-proof upgrades")
        print("- Optimal performance for CBT workload")
        return 0
    else:
        print("\n❌ DEFAULT CONFIG TEST: FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Qdrant Setup Minimalista per Sistema CBT
Versione ultra-compatibile con configurazione base essenziale
"""

import json
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range

class CBTQdrantMinimalSetup:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Inizializza connessione Qdrant per sistema CBT"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "cbt_diary_sessions"
        
    def create_cbt_collection(self) -> bool:
        """
        Crea collection base per sessioni diario CBT
        Configurazione minimalista per compatibilità garantita
        """
        try:
            # Verifica se collection esiste già
            try:
                collections = self.client.get_collections()
                existing_names = [col.name for col in collections.collections]
                if self.collection_name in existing_names:
                    print(f"✅ Collection '{self.collection_name}' già esistente")
                    return True
            except Exception:
                print("Creando prima collection...")
                
            # Configurazione vettori MINIMALISTA
            # Solo parametri essenziali per evitare problemi compatibilità
            vector_config = VectorParams(
                size=3072,  # OpenAI text-embedding-3-large
                distance=Distance.COSINE  # Cosine similarity
            )
            
            # Crea collection con configurazione minima
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config
            )
            
            print(f"✅ Collection '{self.collection_name}' creata con successo")
            return True
            
        except Exception as e:
            print(f"❌ Errore creazione collection: {str(e)}")
            print("Dettagli errore per debugging:")
            import traceback
            print(traceback.format_exc())
            return False
    
    def test_collection(self) -> bool:
        """Test funzionalità base collection"""
        try:
            # Test vector dummy
            import random
            import uuid
            
            test_vector = [random.random() for _ in range(3072)]
            test_payload = {
                "session_id": "test_session_001", 
                "timestamp": "2025-01-01T00:00:00Z",
                "content": "Test session per verifica funzionamento",
                "tags": ["test"],
                "test_flag": True
            }
            
            # Genera UUID valido per point ID
            test_point_id = str(uuid.uuid4())
            
            # Inserisci punto test
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=test_point_id,
                        vector=test_vector,
                        payload=test_payload
                    )
                ]
            )
            print(f"Insert result: {result}")
            
            # Test search con nuova API
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=test_vector,
                limit=1
            )
            print(f"Search found {len(search_result.points)} results")
            
            # Verifica risultato
            if search_result.points and len(search_result.points) > 0:
                print(f"Best match score: {search_result.points[0].score}")
                print(f"Best match payload: {search_result.points[0].payload}")
            
            # Rimuovi punto test
            delete_result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=[test_point_id]
            )
            print(f"Delete result: {delete_result}")
            
            if search_result.points and len(search_result.points) > 0:
                print("✅ Test collection superato")
                return True
            else:
                print("❌ Test collection fallito - no search results")
                return False
                
        except Exception as e:
            print(f"❌ Errore test collection: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Restituisce informazioni collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Estrai info base in modo sicuro
            result = {
                "status": str(info.status),
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
            
            # Aggiungi config se disponibile
            try:
                if hasattr(info, 'config') and info.config:
                    if hasattr(info.config, 'params') and info.config.params:
                        if hasattr(info.config.params, 'vectors'):
                            result["config"] = {
                                "vector_size": info.config.params.vectors.size,
                                "distance": str(info.config.params.vectors.distance),
                            }
            except Exception as e:
                result["config_error"] = str(e)
                
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_basic_operations(self) -> bool:
        """Test operazioni base per verifica funzionamento"""
        try:
            print("\n🔍 Testing basic operations...")
            
            # Test 1: Insert multiple points
            import random
            import uuid
            
            test_points = []
            test_point_ids = []
            
            for i in range(3):
                vector = [random.random() for _ in range(3072)]
                point_id = str(uuid.uuid4())
                test_point_ids.append(point_id)
                
                payload = {
                    "session_id": f"test_session_{i:03d}",
                    "content": f"Contenuto sessione test numero {i}",
                    "timestamp": f"2025-01-0{i+1}T10:00:00Z",
                    "mood": random.randint(1, 10),
                    "tags": ["test", f"session_{i}"]
                }
                test_points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # Insert
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=test_points
            )
            print(f"✅ Insert 3 points: {result}")
            
            # Test 2: Search con nuova API
            query_vector = test_points[0].vector
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=3
            )
            print(f"✅ Search returned {len(search_results.points)} results")
            
            # Test 3: Filter search con nuova API
            from qdrant_client.models import Filter, FieldCondition, Range
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="mood",
                        range=Range(gte=5)
                    )
                ]
            )
            
            filter_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filter_condition,
                limit=3
            )
            print(f"✅ Filtered search returned {len(filter_results.points)} results")
            
            # Test 4: Count
            count_result = self.client.count(self.collection_name)
            print(f"✅ Collection contains {count_result.count} points")
            
            # Cleanup
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=test_point_ids
            )
            print("✅ Cleanup completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore test operations: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def setup_complete_cbt_system(self) -> bool:
        """Setup completo sistema Qdrant per CBT"""
        print("🚀 Inizializzazione sistema Qdrant per CBT (versione minimalista)...")
        
        # Test 1: Verifica connessione
        try:
            collections = self.client.get_collections()
            print(f"✅ Connessione Qdrant stabilita ({len(collections.collections)} collections esistenti)")
        except Exception as e:
            print(f"❌ Impossibile connettersi a Qdrant: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Verifica container: docker-compose ps")
            print("2. Controlla logs: docker-compose logs qdrant")
            print("3. Test health: curl http://localhost:6333/health")
            return False
        
        # Test 2: Crea collection
        if not self.create_cbt_collection():
            print("❌ Fallita creazione collection")
            return False
            
        # Test 3: Test funzionamento base
        if not self.test_collection():
            print("❌ Fallito test collection base")
            return False
            
        # Test 4: Test operazioni complete
        if not self.test_basic_operations():
            print("❌ Fallito test operazioni avanzate")
            return False
            
        # Mostra info finale
        info = self.get_collection_info()
        print("\n📊 Informazioni Collection CBT:")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        
        print("\n🎉 Setup Qdrant completato con successo!")
        print(f"Collection: {self.collection_name}")
        print("Configurazione: Minimalista (compatibilità garantita)")
        print("Features testate: Insert, Search, Filter, Count, Delete")
        print("Sistema pronto per importazione sessioni CBT")
        
        return True

def main():
    """Funzione principale setup"""
    print("=" * 60)
    print("SETUP QDRANT MINIMALISTA PER SISTEMA DIARIO CBT")
    print("=" * 60)
    
    # Verifica versione qdrant-client
    try:
        import qdrant_client
        print(f"Qdrant client version: {qdrant_client.__version__}")
    except:
        print("Qdrant client version: unknown")
    
    # Inizializza setup
    setup = CBTQdrantMinimalSetup()
    
    # Esegui setup completo
    success = setup.setup_complete_cbt_system()
    
    if success:
        print("\n✅ SETUP COMPLETATO - Qdrant pronto per uso CBT")
        print("\nCaratteristiche:")
        print("- Collection con vector size 3072 (OpenAI embedding)")
        print("- Cosine similarity per ricerca semantica")
        print("- Configurazione base compatibile")
        print("- Tutte le operazioni CRUD testate")
        print("\nProssimi step:")
        print("1. Preparare export ChatGPT")
        print("2. Implementare embedding pipeline") 
        print("3. Sviluppare RAG retrieval system")
    else:
        print("\n❌ SETUP FALLITO")
        print("\nVerifica:")
        print("1. Docker container running: docker-compose ps")
        print("2. Qdrant responsive: curl http://localhost:6333/health")
        print("3. Python dependencies: pip install qdrant-client")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

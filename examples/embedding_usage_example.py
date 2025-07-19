#!/usr/bin/env python3
"""
Example usage of the CBT Journal embedding providers
Demonstrates integration with vector store and cost tracking
"""

import asyncio
import os
from typing import Any, Dict

from cbt_journal.ai.embedding_providers import EmbeddingProviderFactory
from cbt_journal.rag.vector_store import create_vector_store
from cbt_journal.utils.cost_control import CostControlManager


async def main() -> None:
    """Example usage of embedding providers with CBT vector store"""

    # Initialize cost manager
    cost_manager = CostControlManager()

    # Create vector store
    vector_store = create_vector_store(cost_manager=cost_manager)

    # Check if we have OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found. Using mock provider for demonstration.")
        provider_type = "mock"
        provider_config = {"simulate_latency": 100}
    else:
        print("Using OpenAI embedding provider")
        provider_type = "openai"
        provider_config = {"api_key": api_key}

    # Create embedding provider
    provider = EmbeddingProviderFactory.create_provider(provider_type, **provider_config)

    print(f"Created {provider_type} provider with {provider.get_dimensions()} dimensions")

    # Example CBT session content
    session_content = """
    Today I felt anxious about the presentation at work.
    I noticed my heart racing and my palms getting sweaty.

    Thought: "Everyone will think I'm incompetent"
    Evidence for: I haven't prepared as much as I'd like
    Evidence against: I've given successful presentations before,
                     colleagues have complimented my work

    Balanced thought: "I'm prepared enough and even if it's not perfect,
                     I can learn from this experience"

    Feeling after: Less anxious, more confident (anxiety: 3/10)
    """

    # Generate embedding
    print("\nGenerating embedding...")
    result = await provider.embed_text(session_content)

    if result.success:
        print("✅ Embedding generated successfully!")
        print(f"   - Vector dimensions: {len(result.vector)}")
        print(f"   - Token count: {result.token_count}")
        print(f"   - Cost: ${result.cost_usd:.6f}")
        print(f"   - Processing time: {result.processing_time_ms}ms")
        print(f"   - Model: {result.model_version}")

        # Create session data for vector store
        session_data: Dict[str, Any] = {
            "session_id": "example_session_001",
            "timestamp": "2024-01-15T10:30:00Z",
            "data_source": "manual_entry",
            "content": {
                "session_text": session_content,
                "session_type": "cbt_thought_record",
                "mood_before": {"anxiety": 8, "confidence": 3},
                "mood_after": {"anxiety": 3, "confidence": 7},
                "techniques_used": ["thought_challenging", "evidence_examination"],
            },
            "ai_models": {
                "embedding_provider": provider_type,
                "embedding_model": result.model_version,
                "embedding_cost": result.cost_usd,
                "embedding_tokens": result.token_count,
            },
        }

        # Store in vector database
        print("\nStoring in vector database...")
        try:
            session_id = vector_store.store_session(
                session_data=session_data, embedding=result.vector
            )
            print(f"✅ Session stored with ID: {session_id}")

            # Record API cost
            cost_manager.record_api_cost(
                session_id=session_id,
                api_type="embedding",
                tokens_input=result.token_count,
                tokens_output=0,
                actual_cost=result.cost_usd,
                processing_time_ms=result.processing_time_ms,
            )
            print(f"✅ Cost tracked: ${result.cost_usd:.6f}")

            # Test similarity search
            print("\nTesting similarity search...")
            search_query = "I'm worried about work performance"
            search_result = await provider.embed_text(search_query)

            if search_result.success:
                similar_sessions = vector_store.search_similar(
                    query_embedding=search_result.vector, limit=5, score_threshold=0.5
                )

                print(f"Found {len(similar_sessions)} similar sessions:")
                for i, session in enumerate(similar_sessions, 1):
                    print(
                        f"  {i}. Session {session['session_id']} "
                        f"(similarity: {session['score']:.3f})"
                    )

            # Get collection stats
            print("\nVector store statistics:")
            stats = vector_store.get_collection_stats()
            print(f"  - Collection: {stats['collection_name']}")
            print(f"  - Total points: {stats['points_count']}")
            print(f"  - Status: {stats['status']}")

        except Exception as e:
            print(f"❌ Error storing in vector database: {str(e)}")
            print("   Make sure Qdrant is running and configured correctly")

    else:
        print(f"❌ Embedding generation failed: {result.error_message}")

    # Show provider information
    print("\nProvider Information:")
    info = EmbeddingProviderFactory.get_provider_info(provider_type)
    for key, value in info.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())

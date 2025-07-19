#!/usr/bin/env python3
"""
Embedding Provider Interface for CBT Journal
Abstract interface for interchangeable embedding providers supporting 3072-dimensional vectors
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Standardized embedding result across all providers"""

    vector: List[float]  # Always 3072 dimensions
    token_count: int  # Normalized from provider response
    cost_usd: float  # Calculated using pricing table
    processing_time_ms: int  # Measured locally
    model_version: str  # Provider model identifier
    input_length: int  # Input text character count
    success: bool  # API call success status
    error_message: Optional[str]  # None if success=True, error details if False
    provider_metadata: Dict[str, Any]  # Raw provider response for debugging


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    def __init__(self, **config: Any) -> None:
        """Initialize provider with configuration"""
        self.price_per_million_tokens = self._get_pricing()

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for input text"""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return embedding dimensions (must be 3072)"""
        pass

    @abstractmethod
    def _get_pricing(self) -> float:
        """Return cost per million tokens in USD"""
        pass

    def _calculate_cost(self, token_count: int) -> float:
        """Calculate cost based on token count"""
        return (token_count / 1_000_000) * self.price_per_million_tokens

    def _error_result(self, error: Exception, text: str, start_time: float) -> EmbeddingResult:
        """Create standardized error result"""
        return EmbeddingResult(
            vector=[0.0] * 3072,
            token_count=0,
            cost_usd=0.0,
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_version="unknown",
            input_length=len(text),
            success=False,
            error_message=str(error),
            provider_metadata={},
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-large"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        self.api_key = api_key
        self.model = model
        super().__init__()

    async def embed_text(self, text: str) -> EmbeddingResult:
        start_time = time.time()
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.embeddings.create(model=self.model, input=text)

            processing_time = int((time.time() - start_time) * 1000)
            token_count = response.usage.total_tokens

            return EmbeddingResult(
                vector=response.data[0].embedding,  # Already 3072 dim
                token_count=token_count,
                cost_usd=self._calculate_cost(token_count),
                processing_time_ms=processing_time,
                model_version=self.model,
                input_length=len(text),
                success=True,
                error_message=None,
                provider_metadata=response.model_dump(),
            )
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            return self._error_result(e, text, start_time)

    def get_dimensions(self) -> int:
        return 3072

    def _get_pricing(self) -> float:
        return 0.130  # $0.130 per 1M tokens


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    @staticmethod
    def create_provider(provider_type: str, **config: Any) -> EmbeddingProvider:
        """Create embedding provider by type"""
        providers = {
            "openai": OpenAIEmbeddingProvider,
        }

        if provider_type not in providers:
            raise ValueError(f"Unsupported provider: {provider_type}")

        return providers[provider_type](**config)

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available provider types"""
        return ["openai"]

    @staticmethod
    def get_provider_info(provider_type: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        info = {
            "openai": {
                "name": "OpenAI",
                "model": "text-embedding-3-large",
                "dimensions": 3072,
                "cost_per_million": 0.130,
                "description": "OpenAI's most capable embedding model",
            },
        }

        if provider_type not in info:
            raise ValueError(f"Unknown provider: {provider_type}")

        return info[provider_type]

#!/usr/bin/env python3
"""
Test suite for embedding providers
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cbt_journal.ai.embedding_providers import (
    EmbeddingProviderFactory,
    EmbeddingResult,
    OpenAIEmbeddingProvider,
)


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass"""

    def test_embedding_result_creation(self) -> None:
        """Test creating EmbeddingResult"""
        result = EmbeddingResult(
            vector=[0.1] * 3072,
            token_count=100,
            cost_usd=0.013,
            processing_time_ms=150,
            model_version="test-model",
            input_length=500,
            success=True,
            error_message=None,
            provider_metadata={"test": "data"},
        )

        assert len(result.vector) == 3072
        assert result.token_count == 100
        assert result.cost_usd == 0.013
        assert result.success is True
        assert result.error_message is None


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider"""

    def test_initialization(self) -> None:
        """Test provider initialization"""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-3-large"
        assert provider.get_dimensions() == 3072
        assert provider._get_pricing() == 0.130

    def test_custom_model(self) -> None:
        """Test provider with custom model"""
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="custom-model")
        assert provider.model == "custom-model"

    def test_cost_calculation(self) -> None:
        """Test cost calculation"""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        cost = provider._calculate_cost(100_000)  # 100k tokens
        expected_cost = (100_000 / 1_000_000) * 0.130
        assert cost == expected_cost

    @pytest.mark.asyncio
    async def test_embed_text_success(self) -> None:
        """Test successful embedding generation"""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 3072)]
        mock_response.usage.total_tokens = 50
        mock_response.model_dump.return_value = {"test": "response"}

        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            provider = OpenAIEmbeddingProvider(api_key="test-key")
            result = await provider.embed_text("Test text for embedding")

            assert result.success is True
            assert len(result.vector) == 3072
            assert result.token_count == 50
            assert result.model_version == "text-embedding-3-large"
            assert result.input_length == len("Test text for embedding")
            assert result.error_message is None
            assert result.provider_metadata == {"test": "response"}

    @pytest.mark.asyncio
    async def test_embed_text_failure(self) -> None:
        """Test embedding generation failure"""
        mock_client = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            provider = OpenAIEmbeddingProvider(api_key="test-key")
            result = await provider.embed_text("Test text")

            assert result.success is False
            assert result.vector == [0.0] * 3072
            assert result.token_count == 0
            assert result.cost_usd == 0.0
            assert result.error_message == "API Error"


class TestEmbeddingProviderFactory:
    """Test embedding provider factory"""

    def test_create_openai_provider(self) -> None:
        """Test creating OpenAI provider"""
        provider = EmbeddingProviderFactory.create_provider(
            "openai", api_key="test-key", model="custom-model"
        )

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "custom-model"

    def test_unsupported_provider(self) -> None:
        """Test creating unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            EmbeddingProviderFactory.create_provider("invalid")

    def test_get_available_providers(self) -> None:
        """Test getting available providers"""
        providers = EmbeddingProviderFactory.get_available_providers()
        assert "openai" in providers
        assert len(providers) == 1

    def test_get_provider_info(self) -> None:
        """Test getting provider information"""
        info = EmbeddingProviderFactory.get_provider_info("openai")
        assert info["name"] == "OpenAI"
        assert info["model"] == "text-embedding-3-large"
        assert info["dimensions"] == 3072
        assert info["cost_per_million"] == 0.130

    def test_get_provider_info_invalid(self) -> None:
        """Test getting info for invalid provider"""
        with pytest.raises(ValueError, match="Unknown provider: invalid"):
            EmbeddingProviderFactory.get_provider_info("invalid")


class TestEmbeddingProviderIntegration:
    """Integration tests for embedding providers"""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    async def test_openai_provider_real_api(self) -> None:
        """Test OpenAI provider with real API (requires API key)"""
        api_key = os.getenv("OPENAI_API_KEY")
        provider = OpenAIEmbeddingProvider(api_key=api_key)

        result = await provider.embed_text("This is a test for CBT journal embedding")

        assert result.success is True
        assert len(result.vector) == 3072
        assert result.token_count > 0
        assert result.cost_usd > 0
        assert result.processing_time_ms > 0
        assert "text-embedding-3-large" in result.model_version


@pytest.mark.asyncio
async def test_end_to_end_workflow() -> None:
    """Test complete workflow from factory to embedding"""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 3072)]
    mock_response.usage.total_tokens = 25
    mock_response.model_dump.return_value = {"model": "test"}

    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = mock_client

        # Create provider through factory
        provider = EmbeddingProviderFactory.create_provider("openai", api_key="test-key")

        # Generate embedding
        result = await provider.embed_text("CBT session content here...")

        # Verify result
        assert result.success is True
        assert len(result.vector) == 3072
        assert result.token_count == 25
        assert result.cost_usd == provider._calculate_cost(25)
        assert result.model_version == "text-embedding-3-large"

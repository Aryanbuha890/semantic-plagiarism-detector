"""
test_ai_detector.py
-------------------
Tests for AI-generated text detection functionality.
"""

from unittest.mock import MagicMock, patch

import torch

from src.core.ai_detector import (
    detect_ai_probability,
    detect_ai_probability_batch,
    detect_document_ai_probability,
    detect_documents_ai_probability,
)


def test_detect_ai_probability_empty_text():
    """Test that empty text returns 0.0 probability."""
    result = detect_ai_probability("")
    assert result == 0.0


def test_detect_ai_probability_none():
    """Test that None input returns 0.0 probability."""
    result = detect_ai_probability(None)
    assert result == 0.0


def test_detect_ai_probability_batch_empty():
    """Test that empty list returns empty list."""
    result = detect_ai_probability_batch([])
    assert result == []


def test_detect_document_ai_probability_empty():
    """Test that empty chunks return zero probabilities."""
    result = detect_document_ai_probability([])
    assert result["overall"] == 0.0
    assert result["max"] == 0.0
    assert result["chunk_scores"] == []


def test_detect_documents_ai_probability_empty():
    """Test that empty dict returns empty dict."""
    result = detect_documents_ai_probability({})
    assert result == {}


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer that return proper tensor outputs."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Tokenizer returns a dict-like object with tensor values
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    # Model returns an object with logits as a real tensor
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.8, 0.2]])
    mock_model.return_value = mock_output

    return mock_model, mock_tokenizer


@patch("src.core.ai_detector._get_model_and_tokenizer")
def test_detect_documents_ai_probability_single_doc(mock_get_model):
    """Test AI detection with a single document."""
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()
    mock_get_model.return_value = (mock_model, mock_tokenizer)

    chunked_docs = {
        "test_doc.txt": ["This is a test chunk of text.", "Another test chunk here."]
    }
    result = detect_documents_ai_probability(chunked_docs)

    assert "test_doc.txt" in result
    assert "overall" in result["test_doc.txt"]
    assert "max" in result["test_doc.txt"]
    assert "chunk_scores" in result["test_doc.txt"]
    assert len(result["test_doc.txt"]["chunk_scores"]) == 2
    assert 0.0 <= result["test_doc.txt"]["overall"] <= 1.0
    assert 0.0 <= result["test_doc.txt"]["max"] <= 1.0


@patch("src.core.ai_detector._get_model_and_tokenizer")
def test_detect_ai_probability_batch_mixed(mock_get_model):
    """Test batch detection with mixed empty and non-empty texts."""
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()
    mock_get_model.return_value = (mock_model, mock_tokenizer)

    texts = ["Some text", "", None, "More text"]
    result = detect_ai_probability_batch(texts)

    assert len(result) == 4
    assert result[1] == 0.0  # Empty string
    assert result[2] == 0.0  # None
    assert 0.0 <= result[0] <= 1.0
    assert 0.0 <= result[3] <= 1.0

"""Tests for chat validation service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

import pytest

from services.chat_validator import ChatValidator, ResponseValidator


class TestResponseValidator:
    """Tests for ResponseValidator Pydantic model."""
    
    def test_response_validator_valid_data(self):
        """Test ResponseValidator with valid data."""
        validator = ResponseValidator(should_continue=True, reason="")
        assert validator.should_continue is True
        assert validator.reason == ""
    
    def test_response_validator_invalid_data(self):
        """Test ResponseValidator with invalid question data."""
        validator = ResponseValidator(should_continue=False, reason="Question not related to experiments")
        assert validator.should_continue is False
        assert validator.reason == "Question not related to experiments"


class TestChatValidator:
    """Tests for ChatValidator service."""
    
    def test_validate_question_valid_experiment_question(self):
        """Test that valid experiment questions pass validation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        should_continue, reason = validator.validate_question(
            "What was the effect of the intervention?"
        )
        
        assert should_continue is True
        assert reason == ""
        mock_client.responses.parse.assert_called_once()
    
    def test_validate_question_invalid_unrelated_question(self):
        """Test that unrelated questions fail validation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": False,
            "reason": "This question is about weather, not experiment analysis."
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        should_continue, reason = validator.validate_question(
            "What's the weather today?"
        )
        
        assert should_continue is False
        assert "weather" in reason.lower()
        mock_client.responses.parse.assert_called_once()
    
    def test_validate_question_empty_string(self):
        """Test that empty questions are rejected."""
        mock_client = MagicMock()
        validator = ChatValidator(mock_client)
        
        should_continue, reason = validator.validate_question("")
        
        assert should_continue is False
        assert "empty" in reason.lower()
        # Should not call API for empty strings
        mock_client.responses.parse.assert_not_called()
    
    def test_validate_question_whitespace_only(self):
        """Test that whitespace-only questions are rejected."""
        mock_client = MagicMock()
        validator = ChatValidator(mock_client)
        
        should_continue, reason = validator.validate_question("   \n\t  ")
        
        assert should_continue is False
        assert "empty" in reason.lower()
        mock_client.responses.parse.assert_not_called()
    
    def test_validate_question_api_error_fail_open(self):
        """Test that API errors fail open (allow question through)."""
        # Mock OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_client.responses.parse.side_effect = Exception("API Error")
        
        validator = ChatValidator(mock_client)
        should_continue, reason = validator.validate_question(
            "What was the causal effect?"
        )
        
        # Should fail open (allow the question)
        assert should_continue is True
        assert reason == ""
    
    def test_validate_question_long_question(self):
        """Test validation with very long questions."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        long_question = "Can you explain " + "the intervention effect " * 100
        
        should_continue, reason = validator.validate_question(long_question)
        
        assert should_continue is True
        mock_client.responses.parse.assert_called_once()
    
    def test_validate_question_special_characters(self):
        """Test validation with special characters."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        should_continue, reason = validator.validate_question(
            "What's the p-value? Is it < 0.05?"
        )
        
        assert should_continue is True
        mock_client.responses.parse.assert_called_once()
    
    def test_validate_question_statistical_terms(self):
        """Test that statistical questions pass validation."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        should_continue, reason = validator.validate_question(
            "Can you explain the confidence interval?"
        )
        
        assert should_continue is True
        assert reason == ""
    
    def test_validate_question_calls_correct_model(self):
        """Test that validation uses the correct model."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        mock_client.responses.parse.return_value = mock_response
        
        validator = ChatValidator(mock_client)
        validator.validate_question("What was the effect?")
        
        # Check that it called with the correct model
        call_args = mock_client.responses.parse.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert call_args[1]["text_format"] == ResponseValidator


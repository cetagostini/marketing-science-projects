"""Tests for model selection functionality."""

from __future__ import annotations

import json
from unittest.mock import Mock, MagicMock
import pytest

from services.model_selector import (
    ModelSelector,
    ModelSelectionError,
    QuasiExperimentSelector,
    Step,
    select_model,
)


class TestModelSelector:
    """Test suite for ModelSelector class."""
    
    def test_valid_synthetic_control_selection(self):
        """Test successful selection of SyntheticControl model."""
        # Mock client response
        mock_client = Mock()
        mock_response = Mock()
        
        # Create valid response
        response_data = {
            "steps": [
                {
                    "explanation": "Analyzing the use case",
                    "output": "Multiple control units available"
                },
                {
                    "explanation": "Choosing model",
                    "output": "SyntheticControl is best for this case"
                }
            ],
            "final_answer": "cp.SyntheticControl"
        }
        mock_response.output_text = json.dumps(response_data)
        mock_client.responses.parse.return_value = mock_response
        
        # Test
        selector = ModelSelector(mock_client)
        model_string, reasoning = selector.select_model("Test experiment description")
        
        # Assertions
        assert model_string == "cp.SyntheticControl"
        assert len(reasoning) == 2
        assert reasoning[0]["explanation"] == "Analyzing the use case"
        assert mock_client.responses.parse.called
    
    def test_valid_interrupted_time_series_selection(self):
        """Test successful selection of InterruptedTimeSeries model."""
        # Mock client response
        mock_client = Mock()
        mock_response = Mock()
        
        response_data = {
            "steps": [
                {
                    "explanation": "Single time series detected",
                    "output": "No control units available"
                },
                {
                    "explanation": "Selecting ITS model",
                    "output": "InterruptedTimeSeries is appropriate"
                }
            ],
            "final_answer": "cp.InterruptedTimeSeries"
        }
        mock_response.output_text = json.dumps(response_data)
        mock_client.responses.parse.return_value = mock_response
        
        # Test
        selector = ModelSelector(mock_client)
        model_string, reasoning = selector.select_model("Single metric analysis")
        
        # Assertions
        assert model_string == "cp.InterruptedTimeSeries"
        assert len(reasoning) == 2
    
    def test_retry_on_invalid_model(self):
        """Test retry logic when LLM returns invalid model."""
        mock_client = Mock()
        
        # First response: invalid model
        invalid_response = Mock()
        invalid_data = {
            "steps": [{"explanation": "test", "output": "test"}],
            "final_answer": "cp.InvalidModel"
        }
        invalid_response.output_text = json.dumps(invalid_data)
        
        # Second response: valid model
        valid_response = Mock()
        valid_data = {
            "steps": [{"explanation": "retry", "output": "corrected"}],
            "final_answer": "cp.SyntheticControl"
        }
        valid_response.output_text = json.dumps(valid_data)
        
        # Configure mock to return different responses
        mock_client.responses.parse.side_effect = [invalid_response, valid_response]
        
        # Test
        selector = ModelSelector(mock_client, max_retries=3)
        model_string, reasoning = selector.select_model("Test")
        
        # Assertions
        assert model_string == "cp.SyntheticControl"
        assert mock_client.responses.parse.call_count == 2
    
    def test_failure_after_max_retries(self):
        """Test that ModelSelectionError is raised after max retries."""
        mock_client = Mock()
        
        # All responses are invalid
        invalid_response = Mock()
        invalid_data = {
            "steps": [{"explanation": "test", "output": "test"}],
            "final_answer": "cp.InvalidModel"
        }
        invalid_response.output_text = json.dumps(invalid_data)
        mock_client.responses.parse.return_value = invalid_response
        
        # Test
        selector = ModelSelector(mock_client, max_retries=3)
        
        with pytest.raises(ModelSelectionError):
            selector.select_model("Test")
        
        assert mock_client.responses.parse.call_count == 3
    
    def test_json_decode_error_handling(self):
        """Test handling of malformed JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "not valid json"
        mock_client.responses.parse.return_value = mock_response
        
        selector = ModelSelector(mock_client, max_retries=1)
        
        with pytest.raises(ModelSelectionError, match="not valid JSON"):
            selector.select_model("Test")
    
    def test_validation_error_handling(self):
        """Test handling of response that doesn't match schema."""
        mock_client = Mock()
        mock_response = Mock()
        
        # Missing required fields
        invalid_schema = {
            "steps": [],
            # missing final_answer
        }
        mock_response.output_text = json.dumps(invalid_schema)
        mock_client.responses.parse.return_value = mock_response
        
        selector = ModelSelector(mock_client, max_retries=1)
        
        with pytest.raises(ModelSelectionError, match="schema validation failed"):
            selector.select_model("Test")
    
    def test_convenience_function(self):
        """Test the convenience select_model function."""
        mock_client = Mock()
        mock_response = Mock()
        
        response_data = {
            "steps": [{"explanation": "test", "output": "test"}],
            "final_answer": "cp.SyntheticControl"
        }
        mock_response.output_text = json.dumps(response_data)
        mock_client.responses.parse.return_value = mock_response
        
        # Test convenience function
        model_string, reasoning = select_model(mock_client, "Test message")
        
        assert model_string == "cp.SyntheticControl"
        assert len(reasoning) == 1


class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_step_model(self):
        """Test Step model creation."""
        step = Step(explanation="test explanation", output="test output")
        assert step.explanation == "test explanation"
        assert step.output == "test output"
    
    def test_quasi_experiment_selector_model(self):
        """Test QuasiExperimentSelector model creation."""
        steps = [
            Step(explanation="step 1", output="output 1"),
            Step(explanation="step 2", output="output 2"),
        ]
        selector = QuasiExperimentSelector(
            steps=steps,
            final_answer="cp.SyntheticControl"
        )
        
        assert len(selector.steps) == 2
        assert selector.final_answer == "cp.SyntheticControl"


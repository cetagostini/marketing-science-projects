"""Integration tests for automatic control selection workflow."""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from services.llm_extraction import LLMExtractor, DataExtractor


@pytest.fixture
def mock_flask_app():
    """Create a mock Flask app with proper context."""
    from flask import Flask
    app = Flask(__name__)
    return app


class TestAutoControlSelectionIntegration:
    """Integration tests for the full auto-selection workflow."""
    
    def test_pipeline_with_auto_selection(self, mock_flask_app):
        """Test full experiment pipeline with automatic control selection."""
        from dash_app import run_experiment_pipeline
        
        # Create realistic dataset
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame({
            "date": dates,
            "DEU": np.random.randn(100) + 100,
            "SWE": np.random.randn(100) + 95,
            "NOR": np.random.randn(100) + 98,
            "DNK": np.random.randn(100) + 97,
        })
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock extraction response (with control_group=None)
        mock_extraction_response = Mock()
        extraction_data = {
            "intervention_start_date": "2024-02-15",
            "intervention_end_date": "2024-03-15",
            "target_var": "DEU",
            "control_group": None,  # This triggers auto-selection
            "extra_variables": None,
        }
        mock_extraction_response.output_text = json.dumps(extraction_data)
        
        # Mock date column extraction
        mock_date_response = Mock()
        mock_date_response.output_text = json.dumps({"date_column": "date"})
        
        # Mock data description
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Dataset contains German and Scandinavian market data"
        
        # Mock data structure identification
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["DEU", "SWE", "NOR", "DNK"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        # Mock model selection
        mock_model_response = Mock()
        mock_model_response.output_text = json.dumps({
            "selected_model": "cp.SyntheticControl",
            "reasoning": ["Market data suitable for synthetic control"],
        })
        
        # Configure mock client to return appropriate responses
        def side_effect_parse(model, input, text_format):
            # Check what's being requested based on text_format
            if text_format.__name__ == "DataExtractor":
                return mock_extraction_response
            elif text_format.__name__ == "DateColumnExtractor":
                return mock_date_response
            elif text_format.__name__ == "DataIdentifier":
                return mock_struct_response
            elif text_format.__name__ == "ModelSelection":
                return mock_model_response
            return Mock()
        
        mock_client.responses.parse.side_effect = side_effect_parse
        mock_client.responses.create.return_value = mock_desc_response
        
        # Create LLM extractor
        llm = LLMExtractor(mock_client)
        
        # Run the pipeline within app context
        message = "Test if intervention on DEU market was effective from Feb 15 to Mar 15, 2024"
        
        with mock_flask_app.app_context():
            result = run_experiment_pipeline(
                llm=llm,
                message=message,
                dataset=df,
            )
        
        # Verify auto-selection artifacts are present
        assert "artifacts" in result
        assert "auto_control_selection" in result["artifacts"]
        
        auto_artifacts = result["artifacts"]["auto_control_selection"]
        assert auto_artifacts is not None
        assert "selected_columns" in auto_artifacts
        assert "beta_statistics" in auto_artifacts
        assert "data_description" in auto_artifacts
        assert "data_structure" in auto_artifacts
        
        # Verify selected columns are from the dataset (excluding target)
        selected = auto_artifacts["selected_columns"]
        assert isinstance(selected, list)
        assert len(selected) > 0
        assert "DEU" not in selected  # Target should not be in controls
        
        # Verify the experiment actually ran
        assert "series" in result
        assert "summary_mean" in result
    
    def test_pipeline_with_manual_controls(self, mock_flask_app):
        """Test that existing manual control flow is unchanged."""
        from dash_app import run_experiment_pipeline
        
        # Create dataset
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame({
            "date": dates,
            "target": np.random.randn(100) + 100,
            "control1": np.random.randn(100) + 95,
            "control2": np.random.randn(100) + 98,
        })
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock extraction response (with manual control_group)
        mock_extraction_response = Mock()
        extraction_data = {
            "intervention_start_date": "2024-02-15",
            "intervention_end_date": "2024-03-15",
            "target_var": "target",
            "control_group": ["control1", "control2"],  # Manual controls
            "extra_variables": None,
        }
        mock_extraction_response.output_text = json.dumps(extraction_data)
        
        # Mock date column extraction
        mock_date_response = Mock()
        mock_date_response.output_text = json.dumps({"date_column": "date"})
        
        # Mock model selection
        mock_model_response = Mock()
        mock_model_response.output_text = json.dumps({
            "selected_model": "cp.SyntheticControl",
            "reasoning": ["Test reasoning"],
        })
        
        def side_effect_parse(model, input, text_format):
            if text_format.__name__ == "DataExtractor":
                return mock_extraction_response
            elif text_format.__name__ == "DateColumnExtractor":
                return mock_date_response
            elif text_format.__name__ == "ModelSelection":
                return mock_model_response
            return Mock()
        
        mock_client.responses.parse.side_effect = side_effect_parse
        
        llm = LLMExtractor(mock_client)
        
        message = "Test intervention on target using control1 and control2"
        
        with mock_flask_app.app_context():
            result = run_experiment_pipeline(
                llm=llm,
                message=message,
                dataset=df,
            )
        
        # Verify auto-selection artifacts are None (not used)
        assert "artifacts" in result
        assert result["artifacts"]["auto_control_selection"] is None
        
        # Verify the experiment ran with manual controls
        assert "series" in result
        assert "summary_mean" in result
    
    def test_auto_selection_artifacts_saved(self, mock_flask_app):
        """Verify that auto-selection metadata is properly saved in experiment results."""
        from dash_app import run_experiment_pipeline
        
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=80)
        df = pd.DataFrame({
            "date": dates,
            "market_a": np.random.randn(80) + 100,
            "market_b": np.random.randn(80) + 95,
            "market_c": np.random.randn(80) + 98,
        })
        
        mock_client = MagicMock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock extraction with None controls
        mock_extraction_response = Mock()
        extraction_data = {
            "intervention_start_date": "2024-02-01",
            "intervention_end_date": "2024-02-28",
            "target_var": "market_a",
            "control_group": None,
            "extra_variables": None,
        }
        mock_extraction_response.output_text = json.dumps(extraction_data)
        
        mock_date_response = Mock()
        mock_date_response.output_text = json.dumps({"date_column": "date"})
        
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Market data with three regions"
        
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["market_a", "market_b", "market_c"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        mock_model_response = Mock()
        mock_model_response.output_text = json.dumps({
            "selected_model": "cp.SyntheticControl",
            "reasoning": ["Market comparison"],
        })
        
        def side_effect_parse(model, input, text_format):
            if text_format.__name__ == "DataExtractor":
                return mock_extraction_response
            elif text_format.__name__ == "DateColumnExtractor":
                return mock_date_response
            elif text_format.__name__ == "DataIdentifier":
                return mock_struct_response
            elif text_format.__name__ == "ModelSelection":
                return mock_model_response
            return Mock()
        
        mock_client.responses.parse.side_effect = side_effect_parse
        mock_client.responses.create.return_value = mock_desc_response
        
        llm = LLMExtractor(mock_client)
        message = "Analyze market_a intervention"
        
        with mock_flask_app.app_context():
            result = run_experiment_pipeline(
                llm=llm,
                message=message,
                dataset=df,
            )
        
        # Deep verify artifact structure
        auto_selection = result["artifacts"]["auto_control_selection"]
        
        # Check all required keys are present
        required_keys = ["selected_columns", "beta_statistics", "data_description", "data_structure"]
        for key in required_keys:
            assert key in auto_selection, f"Missing key: {key}"
        
        # Verify beta_statistics format
        beta_stats = auto_selection["beta_statistics"]
        assert isinstance(beta_stats, dict)
        
        # Each selected column should have statistics
        for col in auto_selection["selected_columns"]:
            assert col in beta_stats
            assert "mean" in beta_stats[col]
            assert "lower_hdi" in beta_stats[col]
            assert "upper_hdi" in beta_stats[col]
        
        # Verify data structure matches DataIdentifier schema
        data_struct = auto_selection["data_structure"]
        assert "date_column" in data_struct
        assert "numerical_columns" in data_struct
        assert data_struct["date_column"] == "date"
    
    def test_auto_selection_error_propagation(self, mock_flask_app):
        """Test that errors in auto-selection are properly propagated."""
        from dash_app import run_experiment_pipeline
        from services.llm_extraction import ExtractionError
        
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "target": np.random.randn(50),
        })
        
        mock_client = MagicMock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock extraction with None controls
        mock_extraction_response = Mock()
        extraction_data = {
            "intervention_start_date": "2024-02-01",
            "intervention_end_date": "2024-02-15",
            "target_var": "target",
            "control_group": None,
            "extra_variables": None,
        }
        mock_extraction_response.output_text = json.dumps(extraction_data)
        
        mock_date_response = Mock()
        mock_date_response.output_text = json.dumps({"date_column": "date"})
        
        # Make data description fail
        mock_client.responses.create.side_effect = Exception("LLM service unavailable")
        
        def side_effect_parse(model, input, text_format):
            if text_format.__name__ == "DataExtractor":
                return mock_extraction_response
            elif text_format.__name__ == "DateColumnExtractor":
                return mock_date_response
            return Mock()
        
        mock_client.responses.parse.side_effect = side_effect_parse
        
        llm = LLMExtractor(mock_client)
        message = "Test intervention"
        
        # Should raise ExtractionError with auto-selection message
        with mock_flask_app.app_context():
            with pytest.raises(ExtractionError, match="Automatic control selection failed"):
                run_experiment_pipeline(
                    llm=llm,
                    message=message,
                    dataset=df,
                )


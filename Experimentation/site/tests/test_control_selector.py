"""Unit tests for automatic control selection service."""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from services.control_selector import (
    ControlSelector,
    ControlSelectionError,
    DataIdentifier,
    FeatureSelector,
)


class TestDataIdentifier:
    """Test DataIdentifier Pydantic model."""
    
    def test_data_identifier_schema_valid(self):
        """Test that DataIdentifier validates correctly with valid data."""
        data = {
            "date_column": "date",
            "numerical_columns": ["col1", "col2", "col3"],
            "categorical_columns": ["cat1"],
        }
        identifier = DataIdentifier.model_validate(data)
        assert identifier.date_column == "date"
        assert identifier.numerical_columns == ["col1", "col2", "col3"]
        assert identifier.categorical_columns == ["cat1"]
    
    def test_data_identifier_schema_optional_categorical(self):
        """Test that categorical_columns is optional."""
        data = {
            "date_column": "date",
            "numerical_columns": ["col1", "col2"],
        }
        identifier = DataIdentifier.model_validate(data)
        assert identifier.categorical_columns is None
    
    def test_data_identifier_schema_missing_required(self):
        """Test that missing required fields raise ValidationError."""
        from pydantic import ValidationError
        
        data = {
            "numerical_columns": ["col1", "col2"],
        }
        with pytest.raises(ValidationError):
            DataIdentifier.model_validate(data)


class TestFeatureSelector:
    """Test FeatureSelector class."""
    
    def test_feature_selector_basic(self):
        """Test basic feature selection with synthetic data."""
        # Create synthetic data where col1 and col2 are predictive, col3 is not
        np.random.seed(42)
        n_samples = 100
        
        col1 = np.random.randn(n_samples)
        col2 = np.random.randn(n_samples)
        col3 = np.random.randn(n_samples) * 0.01  # Very weak signal
        
        # Target is a linear combination of col1 and col2
        y = 2 * col1 + 3 * col2 + np.random.randn(n_samples) * 0.1
        
        x_train = pd.DataFrame({
            "col1": col1,
            "col2": col2,
            "col3": col3,
        })
        y_train = pd.Series(y)
        date_values = pd.date_range("2024-01-01", periods=n_samples).values
        
        selector = FeatureSelector()
        selected_columns, beta_stats = selector.select_features(
            x_train=x_train,
            y_train=y_train,
            date_column_values=date_values,
        )
        
        # Should select col1 and col2, likely not col3
        assert isinstance(selected_columns, list)
        assert len(selected_columns) >= 2  # At minimum col1 and col2
        assert "col1" in selected_columns or "col2" in selected_columns
        
        # Check beta statistics structure
        assert isinstance(beta_stats, dict)
        assert "col1" in beta_stats
        assert "mean" in beta_stats["col1"]
        assert "lower_hdi" in beta_stats["col1"]
        assert "upper_hdi" in beta_stats["col1"]
    
    def test_feature_selector_scaling(self):
        """Test that feature selector properly scales data."""
        np.random.seed(42)
        n_samples = 50
        
        # Create data with very different scales
        x_train = pd.DataFrame({
            "small": np.random.randn(n_samples) * 0.1,
            "large": np.random.randn(n_samples) * 1000,
        })
        y_train = pd.Series(np.random.randn(n_samples))
        date_values = pd.date_range("2024-01-01", periods=n_samples).values
        
        selector = FeatureSelector()
        
        # Should not raise error despite different scales
        selected_columns, beta_stats = selector.select_features(
            x_train=x_train,
            y_train=y_train,
            date_column_values=date_values,
        )
        
        # Verify scalers were fitted
        assert selector.x_scaler is not None
        assert selector.y_scaler is not None
        
        # Check that results are valid
        assert isinstance(selected_columns, list)
        assert isinstance(beta_stats, dict)


class TestControlSelector:
    """Test ControlSelector main orchestrator."""
    
    def test_control_selector_full_workflow(self):
        """Test end-to-end workflow with mocked LLM."""
        # Create mock dataset
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "target": np.random.randn(100),
            "control1": np.random.randn(100),
            "control2": np.random.randn(100),
        })
        
        # Mock OpenAI client
        mock_client = MagicMock()
        
        # Mock data description response
        mock_desc_response = Mock()
        mock_desc_response.output_text = "This dataset contains date, target, control1, and control2 columns."
        
        # Mock data structure identification response
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["target", "control1", "control2"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        # Configure mock to return different responses
        mock_client.responses.create.return_value = mock_desc_response
        mock_client.responses.parse.return_value = mock_struct_response
        
        selector = ControlSelector(mock_client)
        
        result = selector.select_controls(
            df=df,
            target_var="target",
            intervention_start_date="2024-02-15",
        )
        
        # Verify result structure
        assert "selected_columns" in result
        assert "beta_statistics" in result
        assert "data_description" in result
        assert "data_structure" in result
        
        assert isinstance(result["selected_columns"], list)
        assert isinstance(result["beta_statistics"], dict)
        assert isinstance(result["data_description"], str)
        assert isinstance(result["data_structure"], dict)
        
        # Verify LLM was called
        assert mock_client.responses.create.called
        assert mock_client.responses.parse.called
    
    def test_control_selector_error_handling_missing_columns(self):
        """Test error handling when target variable is missing."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "control1": np.random.randn(50),
        })
        
        mock_client = MagicMock()
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Dataset description"
        
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["control1"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        mock_client.responses.create.return_value = mock_desc_response
        mock_client.responses.parse.return_value = mock_struct_response
        
        selector = ControlSelector(mock_client)
        
        # Should raise error because target_var is not in numerical_columns
        with pytest.raises(ControlSelectionError, match="Target variable 'missing_target' not found in numerical columns"):
            selector.select_controls(
                df=df,
                target_var="missing_target",
                intervention_start_date="2024-01-15",
            )
    
    def test_control_selector_error_no_pre_intervention_data(self):
        """Test error handling when intervention date is before all data."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-02-01", periods=50),
            "target": np.random.randn(50),
            "control1": np.random.randn(50),
        })
        
        mock_client = MagicMock()
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Dataset description"
        
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["target", "control1"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        mock_client.responses.create.return_value = mock_desc_response
        mock_client.responses.parse.return_value = mock_struct_response
        
        selector = ControlSelector(mock_client)
        
        # Should raise error because intervention_start_date is before data starts
        with pytest.raises(ControlSelectionError, match="No pre-intervention data available"):
            selector.select_controls(
                df=df,
                target_var="target",
                intervention_start_date="2024-01-01",
            )
    
    def test_control_selector_artifacts(self):
        """Test that artifacts are properly structured and contain expected data."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=80),
            "target": np.random.randn(80),
            "control1": np.random.randn(80),
            "control2": np.random.randn(80),
        })
        
        mock_client = MagicMock()
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Detailed data description"
        
        mock_struct_response = Mock()
        data_identifier = {
            "date_column": "date",
            "numerical_columns": ["target", "control1", "control2"],
            "categorical_columns": None,
        }
        mock_struct_response.output_text = json.dumps(data_identifier)
        
        mock_client.responses.create.return_value = mock_desc_response
        mock_client.responses.parse.return_value = mock_struct_response
        
        selector = ControlSelector(mock_client)
        
        result = selector.select_controls(
            df=df,
            target_var="target",
            intervention_start_date="2024-02-01",
        )
        
        # Verify data_description is saved
        assert result["data_description"] == "Detailed data description"
        
        # Verify data_structure matches what we sent
        assert result["data_structure"]["date_column"] == "date"
        assert "target" in result["data_structure"]["numerical_columns"]
        
        # Verify beta_statistics has proper structure
        for col, stats in result["beta_statistics"].items():
            assert "mean" in stats
            assert "lower_hdi" in stats
            assert "upper_hdi" in stats
            assert isinstance(stats["mean"], (int, float))
            assert isinstance(stats["lower_hdi"], (int, float))
            assert isinstance(stats["upper_hdi"], (int, float))
    
    def test_control_selector_llm_error_handling(self):
        """Test error handling when LLM calls fail."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "target": np.random.randn(50),
            "control1": np.random.randn(50),
        })
        
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("LLM API error")
        
        selector = ControlSelector(mock_client)
        
        with pytest.raises(ControlSelectionError, match="Failed to describe data"):
            selector.select_controls(
                df=df,
                target_var="target",
                intervention_start_date="2024-01-15",
            )
    
    def test_control_selector_invalid_json_response(self):
        """Test error handling when LLM returns invalid JSON."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "target": np.random.randn(50),
            "control1": np.random.randn(50),
        })
        
        mock_client = MagicMock()
        mock_desc_response = Mock()
        mock_desc_response.output_text = "Data description"
        
        mock_struct_response = Mock()
        mock_struct_response.output_text = "Not valid JSON"
        
        mock_client.responses.create.return_value = mock_desc_response
        mock_client.responses.parse.return_value = mock_struct_response
        
        selector = ControlSelector(mock_client)
        
        with pytest.raises(ControlSelectionError, match="Failed to identify data structure"):
            selector.select_controls(
                df=df,
                target_var="target",
                intervention_start_date="2024-01-15",
            )


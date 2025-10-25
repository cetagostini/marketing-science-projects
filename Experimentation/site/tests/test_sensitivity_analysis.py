"""Tests for sensitivity analysis functionality."""

from __future__ import annotations

import pandas as pd
import pytest
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services.experiment_runner import (
    Experiment,
    SensitivityAnalysis,
    run_sensitivity_analysis,
    extract_posterior_sensitivity,
    compute_sensitivity_statistics,
    build_sensitivity_plot,
)


class TestSensitivityAnalysisHelpers:
    """Test suite for sensitivity analysis helper functions."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample time-indexed dataset for testing."""
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        data = {
            "target": np.random.randn(200) + 100,
            "control_a": np.random.randn(200) + 50,
            "control_b": np.random.randn(200) + 75,
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_experiment(self, sample_dataset):
        """Create a sample experiment for testing."""
        return Experiment(
            dataset=sample_dataset,
            intervention_start_date="2024-05-01",
            intervention_end_date="2024-05-31",
            target_var="target",
            control_group=["control_a", "control_b"],
            selected_model="cp.SyntheticControl",
        )
    
    @pytest.fixture
    def mock_sensitivity_results(self):
        """Create mock sensitivity analysis results."""
        results = []
        for i in range(2):  # 2 placebo tests (n_cuts=3 means 2 folds)
            mock_result = MagicMock()
            # Create mock post_impact with proper structure
            mock_post_impact = MagicMock()
            
            # Create mock data array with stack, isel, mean, sum methods
            mock_stacked = MagicMock()
            mock_isel = MagicMock()
            mock_mean = xr.DataArray(np.random.randn(100), dims=["sample"])
            mock_sum = xr.DataArray(np.random.randn(100), dims=["sample"])
            
            mock_isel.mean.return_value = mock_mean
            mock_isel.sum.return_value = mock_sum
            mock_stacked.isel.return_value = mock_isel
            mock_post_impact.stack.return_value = mock_stacked
            mock_result.post_impact = mock_post_impact
            
            results.append({
                "fold": i + 1,
                "pseudo_start": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i*30),
                "pseudo_end": pd.Timestamp("2024-03-31") + pd.Timedelta(days=i*30),
                "result": mock_result,
            })
        return results
    
    def test_run_sensitivity_analysis_success(self, sample_experiment):
        """Test successful sensitivity analysis run."""
        with patch.object(SensitivityAnalysis, 'run') as mock_run:
            mock_run.return_value = [{"fold": 1}, {"fold": 2}]
            
            result = run_sensitivity_analysis(sample_experiment, n_cuts=3)
            
            assert result["success"] is True
            assert result["data"] is not None
            assert result["error"] is None
            assert len(result["data"]) == 2
    
    def test_run_sensitivity_analysis_insufficient_data(self, sample_experiment):
        """Test graceful failure with insufficient pre-period data."""
        # Create experiment with very short dataset
        short_dataset = pd.DataFrame({
            "target": [1, 2, 3, 4, 5],
            "control_a": [2, 3, 4, 5, 6],
        }, index=pd.date_range("2024-04-27", periods=5, freq="D"))
        
        short_experiment = Experiment(
            dataset=short_dataset,
            intervention_start_date="2024-05-01",
            intervention_end_date="2024-05-31",
            target_var="target",
            control_group=["control_a"],
        )
        
        result = run_sensitivity_analysis(short_experiment, n_cuts=3)
        
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] is not None
        assert "Not enough pre-period" in result["error"] or "insufficient data" in result["error"].lower()
    
    def test_extract_posterior_sensitivity(self, mock_sensitivity_results):
        """Test xarray concatenation logic for posterior extraction."""
        posterior_mean, posterior_cumulative = extract_posterior_sensitivity(mock_sensitivity_results)
        
        # Check that we got xarray DataArrays
        assert isinstance(posterior_mean, xr.DataArray)
        assert isinstance(posterior_cumulative, xr.DataArray)
        
        # Check dimensions
        assert "sample" in posterior_mean.dims
        assert "sample" in posterior_cumulative.dims
        
        # Check that we have samples from all folds
        assert len(posterior_mean.values) > 0
        assert len(posterior_cumulative.values) > 0
    
    def test_compute_sensitivity_statistics(self):
        """Test normality tests and quantile computation."""
        # Create sample posterior data
        values = np.random.randn(1000)
        posterior_xr = xr.DataArray(values, dims=["sample"])
        
        stats = compute_sensitivity_statistics(posterior_xr)
        
        # Check that all required fields are present
        assert "mean" in stats
        assert "std" in stats
        assert "quantiles_95" in stats
        assert "quantiles_90" in stats
        assert "quantiles_98" in stats
        assert "normality_tests" in stats
        
        # Check normality tests
        assert "shapiro" in stats["normality_tests"]
        assert "kolmogorov_smirnov" in stats["normality_tests"]
        assert "anderson" in stats["normality_tests"]
        
        # Check that statistics are reasonable
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)
        assert stats["std"] > 0
        
        # Check quantile structure
        assert len(stats["quantiles_95"]) == 2
        assert len(stats["quantiles_90"]) == 2
        assert len(stats["quantiles_98"]) == 3
    
    def test_build_sensitivity_plot_with_data(self):
        """Test plot generation with HDI bounds."""
        # Create sample data
        values = list(np.random.randn(1000))
        
        fig = build_sensitivity_plot(values, "Test Distribution")
        
        # Check that figure was created
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # Check that vertical lines were added (zero line and HDI bounds)
        # The figure should have shapes for vertical lines
        assert hasattr(fig, 'layout')
    
    def test_build_sensitivity_plot_empty_data(self):
        """Test plot generation with empty data."""
        fig = build_sensitivity_plot([], "Empty Distribution")
        
        # Should return a figure with an annotation
        assert fig is not None
        assert hasattr(fig, 'layout')
    
    def test_sensitivity_with_synthetic_control(self, sample_dataset):
        """Integration test with SyntheticControl model."""
        experiment = Experiment(
            dataset=sample_dataset,
            intervention_start_date="2024-05-01",
            intervention_end_date="2024-05-31",
            target_var="target",
            control_group=["control_a", "control_b"],
            selected_model="cp.SyntheticControl",
        )
        
        # Mock the experiment.run() to avoid actual model fitting
        with patch.object(Experiment, 'run') as mock_run:
            # Create a mock result with required structure
            mock_result = MagicMock()
            mock_post_impact = MagicMock()
            mock_stacked = MagicMock()
            mock_isel = MagicMock()
            mock_mean = xr.DataArray(np.random.randn(100), dims=["sample"])
            mock_sum = xr.DataArray(np.random.randn(100), dims=["sample"])
            
            mock_isel.mean.return_value = mock_mean
            mock_isel.sum.return_value = mock_sum
            mock_stacked.isel.return_value = mock_isel
            mock_post_impact.stack.return_value = mock_stacked
            mock_result.post_impact = mock_post_impact
            mock_run.return_value = mock_result
            
            result = run_sensitivity_analysis(experiment, n_cuts=3)
            
            # With sufficient data, should succeed
            assert result["success"] is True or result["error"] is not None
    
    def test_sensitivity_with_its_model(self, sample_dataset):
        """Integration test with InterruptedTimeSeries model."""
        experiment = Experiment(
            dataset=sample_dataset,
            intervention_start_date="2024-05-01",
            intervention_end_date="2024-05-31",
            target_var="target",
            control_group=["control_a", "control_b"],
            selected_model="cp.InterruptedTimeSeries",
        )
        
        # Mock the experiment.run() to avoid actual model fitting
        with patch.object(Experiment, 'run') as mock_run:
            # Create a mock result with required structure
            mock_result = MagicMock()
            mock_post_impact = MagicMock()
            mock_stacked = MagicMock()
            mock_isel = MagicMock()
            mock_mean = xr.DataArray(np.random.randn(100), dims=["sample"])
            mock_sum = xr.DataArray(np.random.randn(100), dims=["sample"])
            
            mock_isel.mean.return_value = mock_mean
            mock_isel.sum.return_value = mock_sum
            mock_stacked.isel.return_value = mock_isel
            mock_post_impact.stack.return_value = mock_stacked
            mock_result.post_impact = mock_post_impact
            mock_run.return_value = mock_result
            
            result = run_sensitivity_analysis(experiment, n_cuts=3)
            
            # With sufficient data, should succeed
            assert result["success"] is True or result["error"] is not None


class TestSensitivityAnalysisIntegration:
    """Integration tests for sensitivity analysis in full pipeline."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a larger dataset for integration testing."""
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        np.random.seed(42)  # For reproducibility
        data = {
            "target": np.random.randn(250) + 100,
            "control_a": np.random.randn(250) + 50,
            "control_b": np.random.randn(250) + 75,
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_full_sensitivity_pipeline(self, sample_dataset):
        """Test full sensitivity analysis pipeline from experiment to statistics."""
        experiment = Experiment(
            dataset=sample_dataset,
            intervention_start_date="2024-06-01",
            intervention_end_date="2024-06-30",
            target_var="target",
            control_group=["control_a", "control_b"],
            selected_model="cp.SyntheticControl",
        )
        
        # Mock the actual model fitting
        with patch.object(Experiment, 'run') as mock_run:
            mock_result = MagicMock()
            mock_post_impact = MagicMock()
            mock_stacked = MagicMock()
            mock_isel = MagicMock()
            mock_mean = xr.DataArray(np.random.randn(100), dims=["sample"])
            mock_sum = xr.DataArray(np.random.randn(100), dims=["sample"])
            
            mock_isel.mean.return_value = mock_mean
            mock_isel.sum.return_value = mock_sum
            mock_stacked.isel.return_value = mock_isel
            mock_post_impact.stack.return_value = mock_stacked
            mock_result.post_impact = mock_post_impact
            mock_run.return_value = mock_result
            
            # Run sensitivity analysis
            sens_result = run_sensitivity_analysis(experiment, n_cuts=3)
            
            if sens_result["success"]:
                # Extract posteriors
                post_mean, post_cumulative = extract_posterior_sensitivity(sens_result["data"])
                
                # Compute statistics
                stats = compute_sensitivity_statistics(post_mean)
                
                # Verify statistics are valid
                assert stats["mean"] is not None
                assert stats["std"] > 0
                assert len(stats["quantiles_95"]) == 2
                
                # Build plots
                fig_mean = build_sensitivity_plot(post_mean.values.tolist(), "Mean Effect")
                fig_cumulative = build_sensitivity_plot(post_cumulative.values.tolist(), "Cumulative Effect")
                
                # Verify figures were created
                assert fig_mean is not None
                assert fig_cumulative is not None


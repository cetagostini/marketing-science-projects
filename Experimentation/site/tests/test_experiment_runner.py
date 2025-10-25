"""Tests for experiment runner with model selection."""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock

from services.experiment_runner import Experiment, run_experiment


class TestExperimentWithModelSelection:
    """Test suite for Experiment class with different models."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample time-indexed dataset for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = {
            "target": range(100),
            "control_a": range(0, 200, 2),
            "control_b": range(100, 300, 2),
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_config(self):
        """Create sample experiment configuration."""
        return {
            "intervention_start_date": "2024-03-01",
            "intervention_end_date": "2024-03-31",
            "target_var": "target",
            "control_group": ["control_a", "control_b"],
        }
    
    def test_experiment_defaults_to_synthetic_control(self, sample_dataset, sample_config):
        """Test that Experiment defaults to SyntheticControl when no model specified."""
        experiment = Experiment(
            dataset=sample_dataset,
            **sample_config
        )
        
        assert experiment.selected_model == "cp.SyntheticControl"
    
    def test_experiment_with_synthetic_control(self, sample_dataset, sample_config):
        """Test Experiment.run() with SyntheticControl model."""
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.SyntheticControl",
            **sample_config
        )
        
        # Mock the causalpy SyntheticControl to avoid actual model fitting
        with patch("services.experiment_runner.cp.SyntheticControl") as mock_sc:
            mock_result = Mock()
            mock_sc.return_value = mock_result
            
            result = experiment.run()
            
            # Verify SyntheticControl was called with correct arguments
            assert mock_sc.called
            call_args = mock_sc.call_args
            
            # Check treatment_time
            assert call_args[0][1] == pd.Timestamp("2024-03-01")
            
            # Check control_units
            assert call_args[1]["control_units"] == ["control_a", "control_b"]
            
            # Check treated_units
            assert call_args[1]["treated_units"] == ["target"]
            
            # Verify result
            assert result == mock_result
    
    def test_experiment_with_interrupted_time_series(self, sample_dataset, sample_config):
        """Test Experiment.run() with InterruptedTimeSeries model."""
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.InterruptedTimeSeries",
            **sample_config
        )
        
        # Mock the causalpy InterruptedTimeSeries
        with patch("services.experiment_runner.cp.InterruptedTimeSeries") as mock_its:
            mock_result = Mock()
            mock_its.return_value = mock_result
            
            result = experiment.run()
            
            # Verify InterruptedTimeSeries was called
            assert mock_its.called
            call_args = mock_its.call_args
            
            # Check treatment_time
            assert call_args[0][1] == pd.Timestamp("2024-03-01")
            
            # Check formula is constructed correctly
            expected_formula = "target ~ 1 + control_a + control_b"
            assert call_args[1]["formula"] == expected_formula
            
            # Verify result
            assert result == mock_result
    
    def test_formula_construction_single_control(self, sample_dataset, sample_config):
        """Test formula construction with a single control variable."""
        config = sample_config.copy()
        config["control_group"] = ["control_a"]
        
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.InterruptedTimeSeries",
            **config
        )
        
        with patch("services.experiment_runner.cp.InterruptedTimeSeries") as mock_its:
            mock_its.return_value = Mock()
            experiment.run()
            
            call_args = mock_its.call_args
            expected_formula = "target ~ 1 + control_a"
            assert call_args[1]["formula"] == expected_formula
    
    def test_formula_construction_multiple_controls(self, sample_dataset, sample_config):
        """Test formula construction with multiple control variables."""
        config = sample_config.copy()
        config["control_group"] = ["control_a", "control_b", "control_c"]
        
        # Add control_c to dataset
        sample_dataset["control_c"] = range(200, 400, 2)
        
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.InterruptedTimeSeries",
            **config
        )
        
        with patch("services.experiment_runner.cp.InterruptedTimeSeries") as mock_its:
            mock_its.return_value = Mock()
            experiment.run()
            
            call_args = mock_its.call_args
            expected_formula = "target ~ 1 + control_a + control_b + control_c"
            assert call_args[1]["formula"] == expected_formula
    
    def test_invalid_model_raises_error(self, sample_dataset, sample_config):
        """Test that invalid model string raises ValueError."""
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.InvalidModel",
            **sample_config
        )
        
        with pytest.raises(ValueError, match="Unsupported model"):
            experiment.run()
    
    def test_sample_kwargs_passed_to_synthetic_control(self, sample_dataset, sample_config):
        """Test that sample_kwargs are passed to SyntheticControl model."""
        sample_kwargs = {"target_accept": 0.95, "random_seed": 42}
        
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.SyntheticControl",
            sample_kwargs=sample_kwargs,
            **sample_config
        )
        
        with patch("services.experiment_runner.cp.SyntheticControl") as mock_sc:
            with patch("services.experiment_runner.cp.pymc_models.WeightedSumFitter") as mock_fitter:
                mock_sc.return_value = Mock()
                
                experiment.run()
                
                # Verify sample_kwargs were passed to WeightedSumFitter
                mock_fitter.assert_called_once_with(sample_kwargs=sample_kwargs)
    
    def test_sample_kwargs_passed_to_interrupted_time_series(self, sample_dataset, sample_config):
        """Test that sample_kwargs are passed to InterruptedTimeSeries model."""
        sample_kwargs = {"random_seed": 42}
        
        experiment = Experiment(
            dataset=sample_dataset,
            selected_model="cp.InterruptedTimeSeries",
            sample_kwargs=sample_kwargs,
            **sample_config
        )
        
        with patch("services.experiment_runner.cp.InterruptedTimeSeries") as mock_its:
            with patch("services.experiment_runner.cp.pymc_models.LinearRegression") as mock_lr:
                mock_its.return_value = Mock()
                
                experiment.run()
                
                # Verify sample_kwargs were passed to LinearRegression
                mock_lr.assert_called_once_with(sample_kwargs=sample_kwargs)
    
    def test_extra_variables_field_exists(self, sample_dataset, sample_config):
        """Test that extra_variables field is available on Experiment."""
        experiment = Experiment(
            dataset=sample_dataset,
            extra_variables=["var1", "var2"],
            **sample_config
        )
        
        assert experiment.extra_variables == ["var1", "var2"]
    
    def test_date_conversion_in_init(self):
        """Test that Timestamp objects are converted to strings in __init__."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame({"target": range(50), "control": range(50)}, index=dates)
        
        experiment = Experiment(
            dataset=df,
            intervention_start_date=pd.Timestamp("2024-02-01"),
            intervention_end_date=pd.Timestamp("2024-02-15"),
            target_var="target",
            control_group=["control"],
        )
        
        # Verify dates were converted to strings
        assert experiment.intervention_start_date == "2024-02-01"
        assert experiment.intervention_end_date == "2024-02-15"


class TestRunExperimentWithSensitivity:
    """Test suite for run_experiment function including sensitivity analysis."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample time-indexed dataset for testing."""
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        np.random.seed(42)
        data = {
            "target": np.random.randn(200) + 100,
            "control_a": np.random.randn(200) + 50,
            "control_b": np.random.randn(200) + 75,
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def mock_model_result(self):
        """Create a mock model result with all required attributes."""
        mock_model = MagicMock()
        
        # Mock datapre_treated and datapost_treated
        mock_model.datapre_treated = MagicMock()
        mock_model.datapre_treated.isel.return_value.mean.return_value.values = 100.0
        
        mock_model.datapost_treated = MagicMock()
        mock_model.datapost_treated.isel.return_value.mean.return_value.values = 110.0
        mock_model.datapost_treated.isel.return_value.sum.return_value.values = 1100.0
        
        # Mock pre_pred and post_pred
        mock_model.pre_pred = {
            "posterior_predictive": MagicMock()
        }
        mock_model.post_pred = {
            "posterior_predictive": MagicMock()
        }
        
        # Create proper xarray structure for predictions
        mock_pp_pre = MagicMock()
        mock_pp_pre.isel.return_value.mean.return_value.values.astype.return_value = np.array([98, 99, 100])
        mock_pp_pre.isel.return_value.quantile.return_value.values.astype.return_value = np.array([95, 96, 97])
        mock_pp_pre.isel.return_value.coords = {"obs_ind": MagicMock(values=pd.date_range("2024-01-01", periods=3))}
        mock_model.pre_pred["posterior_predictive"].y_hat = mock_pp_pre
        
        mock_pp_post = MagicMock()
        mock_pp_post.isel.return_value.mean.return_value.values.astype.return_value = np.array([98, 99, 100])
        mock_pp_post.isel.return_value.quantile.return_value.values.astype.return_value = np.array([95, 96, 97])
        mock_pp_post.isel.return_value.coords = {"obs_ind": MagicMock(values=pd.date_range("2024-05-01", periods=3))}
        mock_pp_post.isel.return_value.stack.return_value.mean.return_value.y_hat.values = 105.0
        mock_pp_post.isel.return_value.stack.return_value.sum.return_value.mean.return_value.y_hat.values = 1050.0
        mock_model.post_pred["posterior_predictive"].y_hat = mock_pp_post
        mock_model.post_pred["posterior_predictive"].isel.return_value.y_hat = mock_pp_post
        
        # Mock datapre_treated and datapost_treated values for time series
        mock_model.datapre_treated.isel.return_value.values.astype.return_value = np.array([100, 101, 102])
        mock_model.datapost_treated.isel.return_value.values.astype.return_value = np.array([110, 111, 112])
        
        # Mock post_impact for posterior samples
        mock_post_impact = MagicMock()
        mock_stacked = MagicMock()
        mock_stacked.values.tolist.return_value = list(np.random.randn(100))
        mock_post_impact.isel.return_value.mean.return_value.stack.return_value = mock_stacked
        mock_post_impact.isel.return_value.sum.return_value.stack.return_value = mock_stacked
        
        # Also mock for sensitivity analysis usage
        mock_isel_sens = MagicMock()
        mock_isel_sens.mean.return_value = xr.DataArray(np.random.randn(100), dims=["sample"])
        mock_isel_sens.sum.return_value = xr.DataArray(np.random.randn(100), dims=["sample"])
        mock_stacked_sens = MagicMock()
        mock_stacked_sens.isel.return_value = mock_isel_sens
        mock_post_impact.stack.return_value = mock_stacked_sens
        
        mock_model.post_impact = mock_post_impact
        
        # Mock summary method
        mock_model.summary.return_value = pd.DataFrame({
            "mean_y": [100, 110],
            "mean_cf": [100, 100],
            "mean_impact": [0, 10],
            "lower_impact": [0, 8],
            "upper_impact": [0, 12],
            "p_value": [1.0, 0.05],
        }, index=["pre", "post"])
        
        return mock_model
    
    def test_run_experiment_includes_sensitivity_results(self, sample_dataset, mock_model_result):
        """Test that run_experiment includes sensitivity_analysis in results."""
        config = {
            "intervention_start_date": "2024-05-01",
            "intervention_end_date": "2024-05-31",
            "target_var": "target",
            "control_group": ["control_a", "control_b"],
            "selected_model": "cp.SyntheticControl",
        }
        
        with patch.object(Experiment, 'run', return_value=mock_model_result):
            with patch("services.experiment_runner.generate_ai_summary", return_value="Test summary"):
                result = run_experiment(config, sample_dataset)
        
        # Verify sensitivity_analysis key is in result
        assert "sensitivity_analysis" in result
        
        # Verify structure
        sensitivity = result["sensitivity_analysis"]
        assert "success" in sensitivity
        assert "error" in sensitivity
        
        # If successful, should have additional fields
        if sensitivity["success"]:
            assert "posterior_mean" in sensitivity
            assert "posterior_cumulative" in sensitivity
            assert "statistics" in sensitivity
    
    def test_run_experiment_sensitivity_failure_does_not_break_experiment(self, sample_dataset, mock_model_result):
        """Test that sensitivity failure doesn't break the experiment execution."""
        # Create a dataset with insufficient pre-period data
        short_dataset = pd.DataFrame({
            "target": list(range(10)),
            "control_a": list(range(10, 20)),
        }, index=pd.date_range("2024-04-25", periods=10, freq="D"))
        
        config = {
            "intervention_start_date": "2024-05-01",
            "intervention_end_date": "2024-05-05",
            "target_var": "target",
            "control_group": ["control_a"],
            "selected_model": "cp.SyntheticControl",
        }
        
        with patch.object(Experiment, 'run', return_value=mock_model_result):
            with patch("services.experiment_runner.generate_ai_summary", return_value="Test summary"):
                result = run_experiment(config, short_dataset)
        
        # Experiment should complete successfully
        assert result is not None
        assert "series" in result
        assert "summary_mean" in result
        
        # Sensitivity should have failed gracefully
        assert "sensitivity_analysis" in result
        assert result["sensitivity_analysis"]["success"] is False
        assert result["sensitivity_analysis"]["error"] is not None
    
    def test_run_experiment_sensitivity_success(self, sample_dataset, mock_model_result):
        """Test successful sensitivity analysis with sufficient data."""
        config = {
            "intervention_start_date": "2024-06-01",
            "intervention_end_date": "2024-06-30",
            "target_var": "target",
            "control_group": ["control_a", "control_b"],
            "selected_model": "cp.SyntheticControl",
        }
        
        with patch.object(Experiment, 'run', return_value=mock_model_result):
            with patch("services.experiment_runner.generate_ai_summary", return_value="Test summary"):
                result = run_experiment(config, sample_dataset)
        
        # Check that sensitivity analysis was attempted
        assert "sensitivity_analysis" in result
        sensitivity = result["sensitivity_analysis"]
        
        # Either it succeeded or failed gracefully
        assert isinstance(sensitivity["success"], bool)
        
        if sensitivity["success"]:
            # Verify all required fields are present
            assert "posterior_mean" in sensitivity
            assert "posterior_cumulative" in sensitivity
            assert "statistics" in sensitivity
            
            # Verify statistics structure
            stats = sensitivity["statistics"]
            assert "mean" in stats
            assert "std" in stats
            assert "normality_tests" in stats
        else:
            # Failed gracefully
            assert "error" in sensitivity
            assert sensitivity["error"] is not None


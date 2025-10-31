"""Experiment orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pydantic import BaseModel
from flask import current_app
from scipy import stats

import causalpy as cp  # type: ignore
import xarray as xr

from services.ai_summary import generate_ai_summary

logger = logging.getLogger(__name__)

try:
    import arviz as az  # type: ignore
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    logger.warning("arviz not available, impact quantiles will not be computed")


class Experiment(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    intervention_start_date: str
    intervention_end_date: str
    target_var: str
    control_group: list[str]
    dataset: pd.DataFrame
    selected_model: str = "cp.SyntheticControl"  # Default to SyntheticControl
    extra_variables: Optional[list[str]] = None
    sample_kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        if "intervention_start_date" in kwargs and hasattr(kwargs["intervention_start_date"], "strftime"):
            kwargs["intervention_start_date"] = kwargs["intervention_start_date"].strftime("%Y-%m-%d")
        if "intervention_end_date" in kwargs and hasattr(kwargs["intervention_end_date"], "strftime"):
            kwargs["intervention_end_date"] = kwargs["intervention_end_date"].strftime("%Y-%m-%d")
        super().__init__(**kwargs)

    def run(self) -> Any:
        """Run the experiment using the selected model.
        
        Returns:
            A causalpy model result (SyntheticControl or InterruptedTimeSeries)
        """
        treatment_time = pd.to_datetime(self.intervention_start_date)
        
        if self.selected_model == "cp.SyntheticControl":
            logger.info("Running SyntheticControl model")
            return cp.SyntheticControl(
                self.dataset,
                treatment_time,
                control_units=self.control_group,
                treated_units=[self.target_var],
                model=cp.pymc_models.WeightedSumFitter(sample_kwargs=self.sample_kwargs),
            )
        elif self.selected_model == "cp.InterruptedTimeSeries":
            logger.info("Running InterruptedTimeSeries model")
            # Build formula: target ~ 1 + control1 + control2 + ...
            formula_parts = [self.target_var, "1"] + self.control_group
            formula = f"{formula_parts[0]} ~ {' + '.join(formula_parts[1:])}"
            logger.info(f"ITS formula: {formula}")
            
            return cp.InterruptedTimeSeries(
                self.dataset,
                treatment_time,
                formula=formula,
                model=cp.pymc_models.LinearRegression(sample_kwargs=self.sample_kwargs),
            )
        else:
            raise ValueError(
                f"Unsupported model: {self.selected_model}. "
                f"Must be one of: cp.SyntheticControl, cp.InterruptedTimeSeries"
            )


class SensitivityAnalysis(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    experiment: Experiment
    n_cuts: int = 2

    def _validate_cuts(self, n_cuts: int) -> None:
        if n_cuts < 2:
            raise ValueError("n_cuts must be >= 2 (n_cuts - 1 folds will be created).")

    def _prepare_pre_data(self, treatment_time: pd.Timestamp) -> pd.DataFrame:
        pre_df = self.experiment.dataset.loc[self.experiment.dataset.index < treatment_time].copy()
        if pre_df.empty:
            raise ValueError("No observations strictly before treatment_time in dataset.")
        return pre_df

    def _calculate_intervention_length(
        self, treatment_time: pd.Timestamp, treatment_time_end: pd.Timestamp
    ) -> pd.Timedelta:
        treatment_time = pd.Timestamp(treatment_time)
        treatment_time_end = pd.Timestamp(treatment_time_end)
        intervention_length = treatment_time_end - treatment_time
        if intervention_length <= pd.Timedelta(0):
            raise ValueError("treatment_time_end must be after treatment_time to compute a positive intervention length.")
        return intervention_length

    def _validate_sufficient_data(
        self,
        pre_df: pd.DataFrame,
        treatment_time: pd.Timestamp,
        intervention_length: pd.Timedelta,
        n_cuts: int,
    ) -> None:
        pre_start = pre_df.index.min()
        earliest_needed = treatment_time - (n_cuts - 1) * intervention_length
        if pre_start > earliest_needed:
            max_cuts = 1 + int((treatment_time - pre_start) // intervention_length)
            raise ValueError(
                "Not enough pre-period for requested folds. "
                f"Earliest required: {earliest_needed.date()}, available starts: {pre_start.date()}. "
                f"Try n_cuts <= {max_cuts}."
            )

    def _create_fold_data(
        self,
        pre_df: pd.DataFrame,
        fold: int,
        n_cuts: int,
        treatment_time: pd.Timestamp,
        intervention_length: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        pseudo_start = treatment_time - (n_cuts - fold) * intervention_length
        pseudo_end = pseudo_start + intervention_length

        fold_df = pre_df.loc[pre_df.index < pseudo_end].sort_index()

        pre_mask = fold_df.index < pseudo_start
        post_mask = (fold_df.index >= pseudo_start) & (fold_df.index < pseudo_end)

        if pre_mask.sum() == 0 or post_mask.sum() == 0:
            raise ValueError(
                f"Fold {fold}: insufficient data. pre_n={pre_mask.sum()}, post_n={post_mask.sum()} "
                f"for window [{pseudo_start} .. {pseudo_end})."
            )

        return fold_df, pseudo_start, pseudo_end

    def _fit_its_model(
        self, fold_df: pd.DataFrame, pseudo_start: pd.Timestamp, pseudo_end: pd.Timestamp
    ) -> Any:
        return Experiment(
            dataset=fold_df,
            intervention_start_date=pseudo_start.strftime("%Y-%m-%d"),
            intervention_end_date=pseudo_end.strftime("%Y-%m-%d"),
            target_var=self.experiment.target_var,
            control_group=self.experiment.control_group,
            selected_model=self.experiment.selected_model,
            extra_variables=self.experiment.extra_variables,
            sample_kwargs=self.experiment.sample_kwargs,
        ).run()

    def run(self) -> list[dict[str, Any]]:
        n_cuts = self.n_cuts
        treatment_time = pd.Timestamp(self.experiment.intervention_start_date)
        treatment_time_end = pd.Timestamp(self.experiment.intervention_end_date)

        self._validate_cuts(n_cuts)
        pre_df = self._prepare_pre_data(treatment_time)
        intervention_length = self._calculate_intervention_length(treatment_time, treatment_time_end)
        self._validate_sufficient_data(pre_df, treatment_time, intervention_length, n_cuts)

        results: list[dict[str, Any]] = []
        for fold in range(1, n_cuts):
            fold_df, pseudo_start, pseudo_end = self._create_fold_data(
                pre_df, fold, n_cuts, treatment_time, intervention_length
            )
            its_result = self._fit_its_model(fold_df, pseudo_start, pseudo_end)
            results.append(
                {
                    "fold": fold,
                    "pseudo_start": pseudo_start,
                    "pseudo_end": pseudo_end,
                    "result": its_result,
                }
            )

        return results


def run_sensitivity_analysis(experiment: Experiment, n_cuts: int = 3) -> Dict[str, Any]:
    """Run sensitivity analysis (placebo testing) for an experiment.
    
    Args:
        experiment: The experiment object to run sensitivity analysis on
        n_cuts: Number of cuts for placebo testing (default 3)
        
    Returns:
        Dictionary with 'success', 'data', and 'error' keys
    """
    try:
        logger.info(f"Running sensitivity analysis with n_cuts={n_cuts}")
        sensitivity_analysis = SensitivityAnalysis(
            experiment=experiment,
            n_cuts=n_cuts,
        )
        sensitivity_results = sensitivity_analysis.run()
        logger.info(f"Sensitivity analysis completed with {len(sensitivity_results)} results")
        return {
            "success": True,
            "data": sensitivity_results,
            "error": None,
        }
    except ValueError as e:
        logger.warning(f"Sensitivity analysis failed (insufficient data): {e}")
        return {
            "success": False,
            "data": None,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Sensitivity analysis failed unexpectedly: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(e)}",
        }


def extract_posterior_sensitivity(sensitivity_results: list) -> Tuple[xr.DataArray, xr.DataArray]:
    """Extract posterior sensitivity distributions from sensitivity analysis results.
    
    Args:
        sensitivity_results: List of sensitivity analysis results from SensitivityAnalysis.run()
        
    Returns:
        Tuple of (posterior_sensitivity_mean, posterior_sensitivity_cumulative)
    """
    logger.info("Extracting posterior sensitivity distributions")
    
    posterior_sensitivity_mean = xr.concat(
        [dat["result"].post_impact.stack(sample=["draw", "chain"]).isel(treated_units=0).mean("obs_ind") 
         for dat in sensitivity_results], 
        dim="sample"
    )
    
    posterior_sensitivity_cumulative = xr.concat(
        [dat["result"].post_impact.stack(sample=["draw", "chain"]).isel(treated_units=0).sum("obs_ind") 
         for dat in sensitivity_results], 
        dim="sample"
    )
    
    logger.info(f"Extracted {len(posterior_sensitivity_mean.values)} posterior samples for mean")
    logger.info(f"Extracted {len(posterior_sensitivity_cumulative.values)} posterior samples for cumulative")
    
    return posterior_sensitivity_mean, posterior_sensitivity_cumulative


def compute_sensitivity_statistics(posterior_sensitivity_mean: xr.DataArray) -> Dict[str, Any]:
    """Compute normality tests and quantiles for sensitivity analysis.
    
    Args:
        posterior_sensitivity_mean: Posterior sensitivity mean distribution
        
    Returns:
        Dictionary containing statistics, normality tests, and quantiles
    """
    logger.info("Computing sensitivity statistics")
    
    values = posterior_sensitivity_mean.values
    
    # Compute normality tests
    shapiro_stat, shapiro_p = stats.shapiro(values)
    ks_stat, ks_p = stats.kstest(values, 'norm', args=(values.mean(), values.std()))
    anderson_result = stats.anderson(values, dist='norm')
    
    # Compute descriptive statistics
    mean_val = float(posterior_sensitivity_mean.mean().item())
    std_val = float(posterior_sensitivity_mean.std().item())
    
    # Compute quantiles
    q_95 = posterior_sensitivity_mean.quantile([0.025, 0.975]).values
    q_90 = posterior_sensitivity_mean.quantile([0.05, 0.95]).values
    q_98 = posterior_sensitivity_mean.quantile([0.01, 0.5, 0.99]).values
    
    statistics = {
        "mean": mean_val,
        "std": std_val,
        "quantiles_95": [float(q_95[0]), float(q_95[1])],
        "quantiles_90": [float(q_90[0]), float(q_90[1])],
        "quantiles_98": [float(q_98[0]), float(q_98[1]), float(q_98[2])],
        "normality_tests": {
            "shapiro": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
            "kolmogorov_smirnov": {"statistic": float(ks_stat), "p_value": float(ks_p)},
            "anderson": {
                "statistic": float(anderson_result.statistic),
                "critical_values": anderson_result.critical_values.tolist(),
                "significance_levels": anderson_result.significance_level.tolist(),
            },
        },
    }
    
    logger.info(f"Sensitivity mean: {mean_val:.4f} ± {std_val:.4f}")
    logger.info(f"95% CI: [{q_95[0]:.4f}, {q_95[1]:.4f}]")
    
    return statistics


def build_sensitivity_plot(posterior_sensitivity_values: list, label: str) -> go.Figure:
    """Build a distribution plot for sensitivity analysis with HDI bounds.
    
    Args:
        posterior_sensitivity_values: List of posterior sensitivity samples
        label: Label for the distribution
        
    Returns:
        Plotly figure showing the distribution with zero line and HDI bounds
    """
    if not posterior_sensitivity_values:
        # Return empty figure if no samples
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=350,
            annotations=[{
                "text": "No sensitivity samples available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14}
            }]
        )
        return fig
    
    try:
        # Convert to xarray for HDI computation
        posterior_xr = xr.DataArray(posterior_sensitivity_values, dims=["sample"])
        
        # Compute HDI bounds
        if HAS_ARVIZ:
            hdi = az.hdi(posterior_xr, hdi_prob=0.84, input_core_dims=[["sample"]])
            hdi_lower = float(hdi.x[0].item())
            hdi_upper = float(hdi.x[1].item())
        else:
            # Fallback to quantiles if arviz not available
            logger.warning("arviz not available, using quantiles instead of HDI")
            hdi_lower = float(posterior_xr.quantile(0.08))
            hdi_upper = float(posterior_xr.quantile(0.92))
        
        # Create distribution plot
        fig = ff.create_distplot(
            [posterior_sensitivity_values],
            [label],
            bin_size=0.01,
            show_rug=False,
            show_hist=True,
            show_curve=True,
        )
        
        # Add vertical line at zero (True results)
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="True results",
            annotation_position="top"
        )
        
        # Add HDI bounds (ROPE)
        fig.add_vline(
            x=hdi_lower,
            line_dash="dash",
            line_color="green",
            annotation_text="Rope lower bound",
            annotation_position="bottom left"
        )
        fig.add_vline(
            x=hdi_upper,
            line_dash="dash",
            line_color="green",
            annotation_text="Rope upper bound",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_white",
            height=350,
            showlegend=False,
            margin=dict(l=40, r=24, t=24, b=40),
            xaxis_title="Effect",
            yaxis_title="Density",
        )
        
        logger.info(f"Built sensitivity plot with HDI [{hdi_lower:.4f}, {hdi_upper:.4f}]")
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to build sensitivity plot: {e}")
        # Return simple histogram as fallback
        fig = go.Figure(data=[go.Histogram(x=posterior_sensitivity_values)])
        fig.update_layout(
            template="plotly_white",
            height=350,
            xaxis_title="Effect",
            yaxis_title="Count",
        )
        return fig


def _to_datetime_string(value: Any) -> str:
    try:
        parsed = pd.to_datetime(value)
    except Exception:  # pragma: no cover - fallback when parsing fails
        return str(value)
    if pd.isna(parsed):
        return str(value)
    return parsed.isoformat()


def _compute_time_series(model: cp.SyntheticControl) -> Dict[str, Dict[str, list[Any]]]:
    datapre = model.datapre_treated.isel(treated_units=0)
    datapost = model.datapost_treated.isel(treated_units=0)

    pp_data_pre = model.pre_pred["posterior_predictive"].y_hat.isel(treated_units=0)
    pp_data_post = model.post_pred["posterior_predictive"].y_hat.isel(treated_units=0)

    pp_mean_pre = pp_data_pre.mean(dim=["draw", "chain"]).values.astype(float)
    pp_lower_pre = pp_data_pre.quantile(0.025, dim=["draw", "chain"]).values.astype(float)
    pp_upper_pre = pp_data_pre.quantile(0.975, dim=["draw", "chain"]).values.astype(float)
    time_pre = [
        _to_datetime_string(value)
        for value in pd.Index(pp_data_pre.coords["obs_ind"].values)
    ]
    observed_pre = [float(x) for x in datapre.values.astype(float)]

    pp_mean_post = pp_data_post.mean(dim=["draw", "chain"]).values.astype(float)
    pp_lower_post = pp_data_post.quantile(0.025, dim=["draw", "chain"]).values.astype(float)
    pp_upper_post = pp_data_post.quantile(0.975, dim=["draw", "chain"]).values.astype(float)
    time_post = [
        _to_datetime_string(value)
        for value in pd.Index(pp_data_post.coords["obs_ind"].values)
    ]
    observed_post = [float(x) for x in datapost.values.astype(float)]

    return {
        "pre": {
            "time": time_pre,
            "mean": pp_mean_pre.tolist(),
            "lower": pp_lower_pre.tolist(),
            "upper": pp_upper_pre.tolist(),
            "observed": observed_pre,
        },
        "post": {
            "time": time_post,
            "mean": pp_mean_post.tolist(),
            "lower": pp_lower_post.tolist(),
            "upper": pp_upper_post.tolist(),
            "observed": observed_post,
        },
    }


def _fallback_summary(experiment: Experiment, dataset: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, float] | None]:
    if experiment.target_var not in dataset.columns:
        return (
            {
                "Actual": {"value": "N/A", "subtitle": None},
                "Estimated": {"value": "N/A", "subtitle": None},
                "Impact": {"value": "N/A", "subtitle": None},
            },
            None,
        )

    target_series = dataset[experiment.target_var].dropna()
    if target_series.empty:
        return (
            {
                "Actual": {"value": "N/A", "subtitle": None},
                "Estimated": {"value": "N/A", "subtitle": None},
                "Impact": {"value": "N/A", "subtitle": None},
            },
            None,
        )

    start = pd.Timestamp(experiment.intervention_start_date)
    end = pd.Timestamp(experiment.intervention_end_date)

    pre = target_series[target_series.index < start]
    post = target_series[(target_series.index >= start) & (target_series.index <= end)]

    if pre.empty or post.empty:
        return (
            {
                "Actual": {"value": "N/A", "subtitle": None},
                "Estimated": {"value": "N/A", "subtitle": None},
                "Impact": {"value": "N/A", "subtitle": None},
            },
            None,
        )

    baseline = float(pre.mean())
    observed = float(post.mean())
    lift = observed - baseline

    return (
        {
            "Actual": {"value": f"{observed:.2f}", "subtitle": None},
            "Estimated": {"value": f"{baseline:.2f}", "subtitle": None},
            "Impact": {"value": f"{lift:+.2f}", "subtitle": "Unavailable"},
        },
        {
            "baseline": baseline,
            "observed": observed,
            "lift": lift,
        },
    )


def _compute_summary_from_model(
    model: cp.SyntheticControl,
    experiment: Experiment,
    dataset: pd.DataFrame,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, float] | None]:
    """Compute summary metrics directly from the model object.
    
    Returns:
        Tuple of (mean_summary_dict, cumulative_summary_dict, stats_dict)
    """
    try:
        # Calculate MEAN metrics from post-treatment data
        actual_mean = float(model.datapost_treated.isel(treated_units=0).mean("obs_ind").values)
        logger.info(f"Actual mean: {actual_mean}")
        
        # Calculate estimated prediction mean
        estimated_mean = float(
            model.post_pred["posterior_predictive"]
            .isel(treated_units=0)
            .stack(sample=["draw", "chain"])
            .mean(["obs_ind", "sample"])
            .y_hat.values
        )
        logger.info(f"Estimated mean: {estimated_mean}")
        
        # Calculate impact statistics with quantiles (using mean, not sum)
        if HAS_ARVIZ:
            impact_summary = az.summary(
                model.post_impact.mean("obs_ind"),
                kind="stats"
            )
            mean_impact = float(impact_summary["mean"].values[0])
            lower_impact = float(impact_summary["hdi_3%"].values[0])
            upper_impact = float(impact_summary["hdi_97%"].values[0])
            logger.info(f"Impact: {mean_impact:.2f} [{lower_impact:.2f}, {upper_impact:.2f}]")
        else:
            # Fallback: compute from post_pred if arviz not available
            impact_data = model.post_pred["posterior_predictive"].isel(treated_units=0).y_hat
            observed_post = model.datapost_treated.isel(treated_units=0)
            impact_samples = (observed_post - impact_data).mean("obs_ind")
            mean_impact = float(impact_samples.mean().values)
            lower_impact = float(impact_samples.quantile(0.03).values)
            upper_impact = float(impact_samples.quantile(0.97).values)
        
        # Calculate CUMULATIVE metrics from post-treatment data
        actual_cumulative = float(model.datapost_treated.isel(treated_units=0).sum("obs_ind").values)
        logger.info(f"Actual cumulative: {actual_cumulative}")
        
        # Calculate estimated prediction cumulative
        estimated_cumulative = float(
            model.post_pred["posterior_predictive"]
            .isel(treated_units=0)
            .stack(sample=["draw", "chain"])
            .sum("obs_ind")
            .mean("sample")
            .y_hat.values
        )
        logger.info(f"Estimated cumulative: {estimated_cumulative}")
        
        # Calculate cumulative impact statistics with quantiles
        if HAS_ARVIZ:
            cumulative_impact_summary = az.summary(
                model.post_impact.sum("obs_ind"),
                kind="stats"
            )
            cumulative_impact = float(cumulative_impact_summary["mean"].values[0])
            cumulative_lower = float(cumulative_impact_summary["hdi_3%"].values[0])
            cumulative_upper = float(cumulative_impact_summary["hdi_97%"].values[0])
            logger.info(f"Cumulative Impact: {cumulative_impact:.2f} [{cumulative_lower:.2f}, {cumulative_upper:.2f}]")
        else:
            # Fallback: compute from post_pred if arviz not available
            impact_data = model.post_pred["posterior_predictive"].isel(treated_units=0).y_hat
            observed_post = model.datapost_treated.isel(treated_units=0)
            cumulative_impact_samples = (observed_post - impact_data).sum("obs_ind")
            cumulative_impact = float(cumulative_impact_samples.mean().values)
            cumulative_lower = float(cumulative_impact_samples.quantile(0.03).values)
            cumulative_upper = float(cumulative_impact_samples.quantile(0.97).values)
        
        mean_summary = {
            "Actual": {
                "value": f"{actual_mean:.2f}",
                "subtitle": None,
            },
            "Estimated": {
                "value": f"{estimated_mean:.2f}",
                "subtitle": None,
            },
            "Impact": {
                "value": f"{mean_impact:+.2f}",
                "subtitle": f"[{lower_impact:+.2f}, {upper_impact:+.2f}]",
            },
        }
        
        cumulative_summary = {
            "Actual": {
                "value": f"{actual_cumulative:.2f}",
                "subtitle": None,
            },
            "Estimated": {
                "value": f"{estimated_cumulative:.2f}",
                "subtitle": None,
            },
            "Impact": {
                "value": f"{cumulative_impact:+.2f}",
                "subtitle": f"[{cumulative_lower:+.2f}, {cumulative_upper:+.2f}]",
            },
        }
        
        stats = {
            "actual": actual_mean,
            "estimated": estimated_mean,
            "mean_impact": mean_impact,
            "lower_impact": lower_impact,
            "upper_impact": upper_impact,
            "actual_cumulative": actual_cumulative,
            "estimated_cumulative": estimated_cumulative,
            "cumulative_impact": cumulative_impact,
            "cumulative_lower": cumulative_lower,
            "cumulative_upper": cumulative_upper,
        }
        
        return (mean_summary, cumulative_summary, stats)
    except Exception as e:
        logger.error(f"Error computing summary from model: {e}")
        # Fall back to old method using impact dataframe
        fallback_mean, fallback_stats = _fallback_summary(experiment, dataset)
        return (fallback_mean, fallback_mean, fallback_stats)


def _compute_summary(
    impact: Any,
    experiment: Experiment,
    dataset: pd.DataFrame,
) -> tuple[Dict[str, Any], Dict[str, float] | None]:
    """Legacy summary computation - kept for backward compatibility."""
    if isinstance(impact, pd.DataFrame) and {"pre", "post"}.issubset(impact.index):
        pre_row = impact.loc["pre"]
        post_row = impact.loc["post"]
        baseline = float(pre_row.mean_y)
        observed = float(post_row.mean_y)
        lift = float(post_row.mean_y - post_row.mean_cf)

        return (
            {
                "Baseline": {
                    "value": f"{baseline:.2f}",
                    "subtitle": None,
                },
                "Observed": {
                    "value": f"{observed:.2f}",
                    "subtitle": None,
                },
                "Lift": {
                    "value": f"{lift:+.2f}",
                    "subtitle": None,
                },
            },
            {
                "baseline": baseline,
                "observed": observed,
                "lift": lift,
                "mean_impact": float(post_row.mean_impact),
                "lower_impact": float(post_row.lower_impact),
                "upper_impact": float(post_row.upper_impact),
                "p_value": float(post_row.p_value),
            },
        )

    return _fallback_summary(experiment, dataset)


def _compute_lift_table(
    impact: Any,
    stats: Dict[str, float] | None,
) -> list[dict[str, Any]]:
    if isinstance(impact, pd.DataFrame) and "post" in impact.index:
        row = impact.loc["post"]
        return [
            {"Metric": "Average Treatment Effect", "Value": f"{row.mean_impact:+.2f}"},
            {
                "Metric": "Credible Interval",
                "Value": f"[{row.lower_impact:+.2f}, {row.upper_impact:+.2f}]",
            },
            {"Metric": "p-value", "Value": f"{row.p_value:.3f}"},
        ]

    lift_value = stats.get("lift") if stats else None
    lift_display = f"{lift_value:+.2f}" if isinstance(lift_value, (int, float)) else "Unavailable"
    return [
        {"Metric": "Average Treatment Effect", "Value": lift_display},
        {"Metric": "Credible Interval", "Value": "Unavailable"},
        {"Metric": "p-value", "Value": "Unavailable"},
    ]


def run_experiment(
    experiment_config: Dict[str, Any],
    dataset: pd.DataFrame,
    sample_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.info("=== run_experiment called ===")
    logger.info(f"Experiment config: {experiment_config}")
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Dataset columns: {dataset.columns.tolist()}")
    logger.info(f"Dataset index type: {type(dataset.index)}")
    
    experiment = Experiment(dataset=dataset, sample_kwargs=sample_kwargs, **experiment_config)
    logger.info(f"Experiment object created with model: {experiment.selected_model}")
    logger.info("Running model...")
    
    model = experiment.run()
    logger.info("Model run completed")

    # Try to compute summary directly from model first
    try:
        summary_mean, summary_cumulative, summary_stats = _compute_summary_from_model(model, experiment, dataset)
        logger.info(f"Summary mean metrics computed from model: {summary_mean}")
        logger.info(f"Summary cumulative metrics computed from model: {summary_cumulative}")
    except Exception as e:
        logger.warning(f"Failed to compute summary from model: {e}, trying legacy method")
        # Fallback to old method using impact dataframe
        try:
            impact = model.summary()
            logger.info(f"Model summary computed: {type(impact)}")
        except Exception as e2:  # pragma: no cover - fallback if causalpy fails to summarise
            logger.warning(f"Failed to compute model summary: {e2}")
            impact = None

        summary_metrics, summary_stats = _compute_summary(impact, experiment, dataset)
        logger.info(f"Summary metrics (legacy): {summary_metrics}")
        # For legacy fallback, use same metrics for both mean and cumulative
        summary_mean = summary_metrics
        summary_cumulative = summary_metrics
    
    # Compute lift table (try to get impact if not already computed)
    try:
        if 'impact' not in locals():
            impact = model.summary()
    except Exception:
        impact = None
    
    lift_table = _compute_lift_table(impact, summary_stats)
    logger.info(f"Lift table: {lift_table}")
    
    time_series = _compute_time_series(model)
    logger.info(f"Time series computed with keys: {time_series.keys()}")

    # Extract posterior samples for distribution plots
    try:
        posterior_mean_effect = model.post_impact.isel(treated_units=0).mean("obs_ind").stack(sample=["draw", "chain"]).values.tolist()
        posterior_cumulative_effect = model.post_impact.isel(treated_units=0).sum("obs_ind").stack(sample=["draw", "chain"]).values.tolist()
        logger.info(f"Extracted {len(posterior_mean_effect)} posterior samples")
    except Exception as e:
        logger.warning(f"Failed to extract posterior samples: {e}")
        posterior_mean_effect = []
        posterior_cumulative_effect = []

    # Run sensitivity analysis (placebo testing)
    logger.info("Starting sensitivity analysis...")
    sensitivity_result = run_sensitivity_analysis(experiment, n_cuts=3)
    
    if sensitivity_result["success"]:
        try:
            posterior_mean, posterior_cumulative = extract_posterior_sensitivity(sensitivity_result["data"])
            stats = compute_sensitivity_statistics(posterior_mean)
            sensitivity_analysis = {
                "success": True,
                "posterior_mean": posterior_mean.values.tolist(),
                "posterior_cumulative": posterior_cumulative.values.tolist(),
                "statistics": stats,
                "error": None,
            }
            logger.info("Sensitivity analysis completed successfully")
        except Exception as e:
            logger.error(f"Failed to process sensitivity results: {e}")
            sensitivity_analysis = {
                "success": False,
                "error": f"Failed to process sensitivity results: {str(e)}",
            }
    else:
        sensitivity_analysis = {
            "success": False,
            "error": sensitivity_result["error"],
        }
        logger.info(f"Sensitivity analysis skipped: {sensitivity_result['error']}")

    result = {
        "series": time_series,
        "summary": summary_mean,  # Keep for backward compatibility
        "summary_mean": summary_mean,
        "summary_cumulative": summary_cumulative,
        "lift_table": lift_table,
        "posterior_mean_effect": posterior_mean_effect,
        "posterior_cumulative_effect": posterior_cumulative_effect,
        "sensitivity_analysis": sensitivity_analysis,
        "intervention_start_date": experiment.intervention_start_date,
        "intervention_end_date": experiment.intervention_end_date,
        "target_var": experiment.target_var,
        "control_group": experiment.control_group,
    }
    
    # Generate AI summary if OpenAI client is available
    try:
        client = current_app.config.get("OPENAI_CLIENT")
        if client is not None:
            logger.info("Generating AI summary...")
            ai_summary = generate_ai_summary(client, result, model)
            result["ai_summary"] = ai_summary
            logger.info(f"AI summary {'generated' if ai_summary else 'failed'}")
        else:
            logger.warning("OpenAI client not available, skipping AI summary")
            result["ai_summary"] = None
    except Exception as e:
        logger.error(f"Failed to generate AI summary: {e}")
        result["ai_summary"] = None
    
    logger.info("run_experiment completed successfully")
    return result


def build_series_chart(series: Any) -> go.Figure:
    fig = go.Figure()

    if isinstance(series, list):
        df = pd.DataFrame(series)
        if {"Date", "Observed", "Counterfactual"}.issubset(df.columns):
            fig.add_trace(
                go.Scatter(x=df["Date"], y=df["Observed"], mode="lines+markers", name="Observed")
            )
            fig.add_trace(
                go.Scatter(x=df["Date"], y=df["Counterfactual"], mode="lines+markers", name="Counterfactual")
            )
        else:
            fig.add_trace(go.Scatter(y=df.iloc[:, 0], mode="lines", name="Series"))
        fig.update_layout(template="plotly_white", height=360)
        return fig

    if not isinstance(series, dict):
        fig.update_layout(template="plotly_white", height=360)
        return fig

    pre = series.get("pre", {})
    post = series.get("post", {})

    if pre:
        fig.add_trace(
            go.Scatter(
                x=pre.get("time"),
                y=pre.get("upper"),
                mode="lines",
                line_color="rgba(0,100,80,0)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pre.get("time"),
                y=pre.get("lower"),
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0)",
                name="95% Credible Interval (Pre)",
                fillcolor="rgba(0,100,80,0.2)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pre.get("time"),
                y=pre.get("mean"),
                mode="lines",
                name="Posterior Predictive Mean (Pre)",
                line=dict(color="rgb(0,100,80)", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pre.get("time"),
                y=pre.get("observed"),
                mode="lines",
                name="Observed Data (Pre)",
                line=dict(color="rgb(255,0,0)", width=2),
            )
        )

    if post:
        fig.add_trace(
            go.Scatter(
                x=post.get("time"),
                y=post.get("upper"),
                mode="lines",
                line_color="rgba(0,0,255,0)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=post.get("time"),
                y=post.get("lower"),
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,255,0)",
                name="95% Credible Interval (Post)",
                fillcolor="rgba(0,0,255,0.2)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=post.get("time"),
                y=post.get("mean"),
                mode="lines",
                name="Posterior Predictive Mean (Post)",
                line=dict(color="rgb(0,0,255)", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=post.get("time"),
                y=post.get("observed"),
                mode="lines",
                name="Observed Data (Post)",
                line=dict(color="rgb(255,100,0)", width=2),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=24, t=24, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_distribution_plot(posterior_samples: list[float], label: str) -> go.Figure:
    """Build a distribution plot from posterior samples.
    
    Args:
        posterior_samples: List of posterior sample values
        label: Label for the distribution
        
    Returns:
        Plotly figure showing the distribution with a zero reference line
    """
    if not posterior_samples:
        # Return empty figure if no samples
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=350,
            annotations=[{
                "text": "No posterior samples available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14}
            }]
        )
        return fig
    
    # Create distribution plot
    fig = ff.create_distplot(
        [posterior_samples],
        [label],
        bin_size=(max(posterior_samples) - min(posterior_samples)) / 50,
        show_rug=False,
        show_hist=True,
        show_curve=True,
    )
    
    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Zero",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False,
        margin=dict(l=40, r=24, t=24, b=40),
        xaxis_title="Effect",
        yaxis_title="Density",
    )
    
    return fig



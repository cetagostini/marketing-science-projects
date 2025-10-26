"""Dash implementation of the experimentation console prototype."""

from __future__ import annotations

from datetime import datetime
import base64
import io
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import (
    Dash,
    Input,
    Output,
    State,
    ALL,
    callback_context,
    dash_table,
    dcc,
    html,
)
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from flask import session, has_request_context, current_app
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from storage import load_experiments, save_experiments, load_planning_sessions, save_planning_sessions
from services.llm_extraction import LLMExtractor, ExtractionError, normalise_control_codes
from services.experiment_runner import run_experiment, build_series_chart, build_distribution_plot, build_sensitivity_plot
from services.model_selector import ModelSelector, ModelSelectionError
from server import server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_popover_position(
    viewport_height: Optional[int],
    composer_midpoint: Optional[int],
    popover_height: int = 260,
) -> str:
    if viewport_height is None or composer_midpoint is None:
        return "attachment-top"

    space_below = max(viewport_height - composer_midpoint, 0)

    if space_below >= popover_height:
        return "attachment-bottom"
    return "attachment-top"


def build_composer_chips(dataset: Optional[str]) -> List[html.Span]:
    chips: List[html.Span] = []
    if dataset:
        chips.append(html.Span(dataset, className="composer-chip"))
    return chips


def build_upload_message(filename: Optional[str]) -> Optional[html.Div]:
    if not filename:
        return None
    return html.Div(
        html.Span(f"Loaded dataset: {filename}", className="upload-status"),
        className="upload-feedback",
    )


def _generate_experiment_name(count: int) -> str:
    return f"Experiment {count}"


def _generate_planning_session_name(count: int) -> str:
    return f"Planning Session {count}"


class ExperimentFailure(Exception):
    """Raised when an experiment cannot be executed."""


def parse_upload(contents: Optional[str], filename: Optional[str]) -> Optional[pd.DataFrame]:
    if not contents or not filename:
        return None

    content_type, _, data = contents.partition(",")
    if not data:
        raise ExperimentFailure("Uploaded file content is invalid.")

    try:
        decoded = base64.b64decode(data)
    except (ValueError, TypeError) as exc:
        raise ExperimentFailure("Unable to decode uploaded file.") from exc

    buffer = io.StringIO(decoded.decode("utf-8"))
    try:
        return pd.read_csv(buffer)
    except Exception as exc:  # pragma: no cover - pandas might raise many variants
        raise ExperimentFailure("Failed to parse uploaded CSV.") from exc


def _default_form_values() -> Dict[str, Any]:
    return {
        "message": "",
        "filename": None,
    }


def run_experiment_pipeline(
    llm: LLMExtractor,
    message: str,
    dataset: pd.DataFrame,
    sample_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.info("=== run_experiment_pipeline called ===")
    logger.info(f"Dataset shape: {dataset.shape}")
    
    try:
        logger.info("Extracting experiment details from message...")
        extracted = llm.extract_experiment_details(message)
        logger.info(f"LLM extracted experiment details successfully")
    except Exception as e:
        logger.error(f"Failed to extract experiment details: {e}")
        logger.error(traceback.format_exc())
        raise ExtractionError(f"Failed to extract experiment parameters: {str(e)}") from e
    
    # Check if control_group is None - trigger automatic selection
    auto_selection_artifacts = None
    if extracted.control_group is None:
        logger.info("No control group specified, triggering automatic selection...")
        client = current_app.config.get("OPENAI_CLIENT")
        if client is None:
            raise ExtractionError("Cannot auto-select controls: OpenAI client not configured")
        
        from services.control_selector import ControlSelector
        selector = ControlSelector(client)
        
        try:
            selection_result = selector.select_controls(
                df=dataset,
                target_var=extracted.target_var,
                intervention_start_date=extracted.intervention_start_date
            )
            control_group = selection_result["selected_columns"]
            auto_selection_artifacts = selection_result
            logger.info(f"Auto-selected control groups: {control_group}")
        except Exception as e:
            logger.error(f"Automatic control selection failed: {e}")
            logger.error(traceback.format_exc())
            raise ExtractionError(f"Automatic control selection failed: {str(e)}") from e
    else:
        control_group = normalise_control_codes(extracted.control_group)
        logger.info(f"Normalised control group: {control_group}")

    try:
        logger.info("Extracting date column from dataset...")
        date_column = llm.extract_date_column(dataset)
        logger.info(f"Identified date column: {date_column}")
    except Exception as e:
        logger.error(f"Failed to extract date column: {e}")
        logger.error(traceback.format_exc())
        raise ExtractionError(f"Failed to identify date column: {str(e)}") from e
    
    indexed_dataset = dataset.set_index(pd.to_datetime(dataset[date_column]))
    logger.info(f"Created indexed dataset with index: {indexed_dataset.index[:5].tolist()}")

    # Select the appropriate model based on user's message
    client = current_app.config.get("OPENAI_CLIENT")
    if client is None:
        logger.warning("OpenAI client not available, defaulting to SyntheticControl")
        selected_model = "cp.SyntheticControl"
        model_reasoning = []
    else:
        try:
            logger.info("Selecting model using LLM...")
            model_selector = ModelSelector(client)
            selected_model, model_reasoning = model_selector.select_model(message)
            logger.info(f"Selected model: {selected_model}")
        except ModelSelectionError as e:
            logger.error(f"Model selection failed: {e}, defaulting to SyntheticControl")
            selected_model = "cp.SyntheticControl"
            model_reasoning = []
        except Exception as e:
            logger.error(f"Unexpected error during model selection: {e}, defaulting to SyntheticControl")
            logger.error(traceback.format_exc())
            selected_model = "cp.SyntheticControl"
            model_reasoning = []

    experiment_config = {
        "intervention_start_date": extracted.intervention_start_date,
        "intervention_end_date": extracted.intervention_end_date,
        "target_var": extracted.target_var,
        "control_group": control_group,
        "selected_model": selected_model,
        "extra_variables": extracted.extra_variables,
    }
    logger.info(f"Experiment config: {experiment_config}")

    outputs = run_experiment(experiment_config, indexed_dataset, sample_kwargs=sample_kwargs)
    
    # Add artifacts to outputs
    outputs["artifacts"] = {
        "selected_model": selected_model,
        "model_selection_reasoning": model_reasoning,
        "auto_control_selection": auto_selection_artifacts,
    }
    
    logger.info("run_experiment_pipeline completed successfully")
    return outputs


def toggle_attachment_area_logic(
    triggered: Optional[str],
    toggle_clicks: Optional[int],
    selection: str,
    *,
    viewport_height: Optional[int] = None,
    composer_y: Optional[int] = None,
):
    if selection != "New Experiment":
        return {"display": "none"}

    if triggered == "composer-send":
        return {"display": "none"}

    show = (toggle_clicks or 0) % 2 == 1
    if not show:
        return {"display": "none"}

    placement = compute_popover_position(viewport_height, composer_y)
    return {"display": "block", "className": f"attachment-popover {placement}"}


def compute_popover_position(
    viewport_height: Optional[int],
    composer_midpoint: Optional[int],
    popover_height: int = 260,
) -> str:
    if viewport_height is None or composer_midpoint is None:
        return "attachment-top"

    space_above = composer_midpoint
    space_below = viewport_height - composer_midpoint

    if space_above >= popover_height:
        return "attachment-top"
    if space_below >= popover_height:
        return "attachment-bottom"
    return "attachment-top"


def _build_posterior_accordion(posterior_mean_effect: List, posterior_cumulative_effect: List) -> Optional[dmc.Accordion]:
    """Build accordion for posterior effect distributions (mean and cumulative)."""
    if not posterior_mean_effect and not posterior_cumulative_effect:
        return None
    
    panel_content = []
    
    if posterior_mean_effect:
        panel_content.extend([
            html.H2("Posterior Distribution of Mean Effect", style={"fontSize": "1.25rem", "fontWeight": "600", "marginTop": "1rem", "marginBottom": "0.5rem"}),
            html.P(
                "This distribution represents the uncertainty in the average daily effect during the intervention period. "
                "Each sample from the posterior distribution represents a plausible value for the mean effect, "
                "given the observed data and model assumptions.",
                style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
            ),
            dcc.Graph(
                figure=build_distribution_plot(posterior_mean_effect, "Posterior mean effect"),
                config={"displayModeBar": False},
                className="result-chart",
            ),
        ])
    
    if posterior_cumulative_effect:
        panel_content.extend([
            html.H2("Posterior Distribution of Cumulative Effect", style={"fontSize": "1.25rem", "fontWeight": "600", "marginTop": "1.5rem", "marginBottom": "0.5rem"}),
            html.P(
                "This distribution represents the uncertainty in the total accumulated effect during the intervention period. "
                "Each sample from the posterior distribution represents a plausible value for the cumulative effect, "
                "which is the sum of all daily effects throughout the intervention.",
                style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
            ),
            dcc.Graph(
                figure=build_distribution_plot(posterior_cumulative_effect, "Posterior cumulative effect"),
                config={"displayModeBar": False},
                className="result-chart",
            ),
        ])
    
    return dmc.Accordion(
        disableChevronRotation=True,
        chevronPosition="left",
        variant="separated",
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Posterior Effect Distributions",
                        icon=DashIconify(
                            icon="material-symbols:data-check-rounded",
                            width=20,
                        ),
                    ),
                    dmc.AccordionPanel(panel_content),
                ],
                value="posterior",
            ),
        ],
        style={"marginTop": "2rem", "marginBottom": "1.5rem"}
    )


def _build_sensitivity_accordion(sensitivity_analysis: Dict[str, Any]) -> Optional[dmc.Accordion]:
    """Build accordion for sensitivity analysis (placebo testing)."""
    if not sensitivity_analysis:
        return None
    
    panel_content = []
    
    if sensitivity_analysis.get("success"):
        # Successfully ran sensitivity analysis - show plots and statistics
        posterior_mean_sens = sensitivity_analysis.get("posterior_mean", [])
        posterior_cumulative_sens = sensitivity_analysis.get("posterior_cumulative", [])
        stats = sensitivity_analysis.get("statistics", {})
        
        panel_content.append(
            html.P(
                "Sensitivity analysis validates the model's robustness by running placebo tests in the pre-intervention period. "
                "The model is applied to time windows before the actual intervention to check if it spuriously detects effects when none should exist. "
                "If the placebo effect distribution is centered near zero and the true effect is outside this range, "
                "it strengthens confidence in the causal interpretation.",
                style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
            )
        )
        
        # Add statistics metrics
        if stats:
            mean_val = stats.get("mean", "N/A")
            std_val = stats.get("std", "N/A")
            q_95 = stats.get("quantiles_95", ["N/A", "N/A"])
            normality = stats.get("normality_tests", {})
            shapiro_p = normality.get("shapiro", {}).get("p_value", "N/A")
            
            stats_cards = [
                html.Div([
                    html.Div("Placebo Mean", className="metric-label"),
                    html.Div(f"{mean_val:.4f}" if isinstance(mean_val, (int, float)) else mean_val, className="metric-value"),
                ], className="metric-card"),
                html.Div([
                    html.Div("Placebo Std Dev", className="metric-label"),
                    html.Div(f"{std_val:.4f}" if isinstance(std_val, (int, float)) else std_val, className="metric-value"),
                ], className="metric-card"),
                html.Div([
                    html.Div("95% CI", className="metric-label"),
                    html.Div(f"[{q_95[0]:.4f}, {q_95[1]:.4f}]" if all(isinstance(x, (int, float)) for x in q_95) else str(q_95), className="metric-value"),
                ], className="metric-card"),
                html.Div([
                    html.Div("Normality (p-value)", className="metric-label"),
                    html.Div(f"{shapiro_p:.4f}" if isinstance(shapiro_p, (int, float)) else shapiro_p, className="metric-value"),
                ], className="metric-card"),
            ]
            
            panel_content.append(html.Div(stats_cards, className="metric-grid"))
        
        # Add mean effect distribution plot
        if posterior_mean_sens:
            panel_content.extend([
                html.H2("Placebo Mean Effect Distribution", style={"fontSize": "1.25rem", "fontWeight": "600", "marginTop": "1.5rem", "marginBottom": "0.5rem"}),
                html.P(
                    "Distribution of mean effects detected during placebo tests. Values near zero indicate the model is not detecting spurious effects.",
                    style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
                ),
                dcc.Graph(
                    figure=build_sensitivity_plot(posterior_mean_sens, "Posterior estimated effect during placebo's test"),
                    config={"displayModeBar": False},
                    className="result-chart",
                ),
            ])
        
        # Add cumulative effect distribution plot
        if posterior_cumulative_sens:
            panel_content.extend([
                html.H2("Placebo Cumulative Effect Distribution", style={"fontSize": "1.25rem", "fontWeight": "600", "marginTop": "1.5rem", "marginBottom": "0.5rem"}),
                html.P(
                    "Distribution of cumulative effects detected during placebo tests. Values near zero indicate the model is not detecting spurious cumulative effects.",
                    style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
                ),
                dcc.Graph(
                    figure=build_sensitivity_plot(posterior_cumulative_sens, "Posterior cumulative effect during placebo's test"),
                    config={"displayModeBar": False},
                    className="result-chart",
                ),
            ])
    else:
        # Sensitivity analysis failed - show error message
        error_message = sensitivity_analysis.get("error", "Unknown error")
        panel_content.append(
            html.Div(
                [
                    html.Div("⚠️ Sensitivity Analysis Unavailable", className="error-notification-icon", style={"fontSize": "1.5rem"}),
                    html.Div(
                        [
                            html.P("Sensitivity analysis could not be estimated:", style={"fontWeight": "600", "marginBottom": "0.5rem"}),
                            html.P(error_message, style={"fontStyle": "italic"}),
                        ],
                        className="error-notification-message"
                    ),
                ],
                className="error-notification",
                style={"marginBottom": "1.5rem"}
            )
        )
    
    return dmc.Accordion(
        disableChevronRotation=True,
        chevronPosition="left",
        variant="separated",
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Sensitivity Analysis (Placebo Testing)",
                        icon=DashIconify(
                            icon="material-symbols:shield-toggle-outline",
                            width=20,
                        ),
                    ),
                    dmc.AccordionPanel(panel_content),
                ],
                value="sensitivity",
            ),
        ],
        style={"marginTop": "2rem", "marginBottom": "1.5rem"}
    )


def _render_experiment_detail(experiment: Dict[str, Any]) -> List[html.Component]:
    start_label = _format_date(experiment.get("start_date"))
    end_label = _format_date(experiment.get("end_date"))
    dataset_label = experiment.get("dataset_name") or "No file attached"
    
    # Extract model name from artifacts
    artifacts = experiment.get("results", {}).get("artifacts", {})
    model_label = _format_model_name(artifacts.get("selected_model"))

    # Helper function to build metric cards from summary dict
    def build_metric_cards(summary_dict):
        cards = []
        for label, metric_data in summary_dict.items():
            # Handle both old format (string) and new format (dict with value/subtitle)
            if isinstance(metric_data, dict):
                value = metric_data.get("value", "N/A")
                subtitle = metric_data.get("subtitle")
            else:
                value = metric_data
                subtitle = None
            
            # Build card content
            card_content = [
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
            ]
            
            # Add subtitle if present
            if subtitle:
                card_content.append(
                    html.Div(subtitle, className="metric-subtitle")
                )
            
            cards.append(
                html.Div(card_content, className="metric-card")
            )
        return cards

    # Build mean metric cards
    summary_mean_cards = build_metric_cards(experiment["results"].get("summary_mean", experiment["results"]["summary"]))
    
    # Build cumulative metric cards (if available)
    summary_cumulative_cards = build_metric_cards(experiment["results"].get("summary_cumulative", {}))

    chart = dcc.Graph(
        figure=build_series_chart(experiment["results"]["series"]),
        config={"displayModeBar": False},
        className="result-chart",
    )

    table = dash_table.DataTable(
        columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
        data=experiment["results"]["lift_table"],
        style_as_list_view=True,
        cell_selectable=False,
        style_cell={"padding": "0.75rem", "border": "none"},
        style_header={"display": "none"},
        style_table={"marginTop": "1rem"},
    )

    components = [
        html.Div(
            [
                html.Div("Request summary", className="section-title"),
                html.Div(
                    [
                        html.Div("Original brief", className="message-title"),
                        html.Div(experiment["message"], className="message-body"),
                    ],
                    className="message-card",
                ),
                html.Div(
                    [
                        html.Div([html.Div("Start date", className="meta-label"), html.Div(start_label, className="meta-value")]),
                        html.Div([html.Div("End date", className="meta-label"), html.Div(end_label, className="meta-value")]),
                        html.Div([html.Div("Dataset", className="meta-label"), html.Div(dataset_label, className="meta-value")]),
                        html.Div([html.Div("Model", className="meta-label"), html.Div(model_label, className="meta-value")]),
                    ],
                    className="meta-grid",
                ),
            ],
            className="detail-card",
        ),
        html.H1("Intervention mean estimation", style={"fontSize": "1.5rem", "fontWeight": "600", "marginTop": "1.5rem", "marginBottom": "0.5rem"}),
        html.P(
            "Average daily effect during the intervention period. These metrics represent the mean difference between observed and predicted values.",
            style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
        ),
        html.Div(summary_mean_cards, className="metric-grid"),
    ]
    
    # Add cumulative section only if cumulative data is available
    if summary_cumulative_cards:
        components.extend([
            html.H1("Intervention cumulative estimation", style={"fontSize": "1.5rem", "fontWeight": "600", "marginTop": "2rem", "marginBottom": "0.5rem"}),
            html.P(
                "Total accumulated effect during the intervention period. These metrics represent the sum of differences between observed and predicted values.",
                style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
            ),
            html.Div(summary_cumulative_cards, className="metric-grid"),
        ])
    
    # Add remaining sections
    components.extend([
        html.H1("Posterior Predictive Time Series", style={"fontSize": "1.5rem", "fontWeight": "600", "marginTop": "2rem", "marginBottom": "0.5rem"}),
        html.P(
            "This visualization shows the observed data alongside the model's posterior predictive estimates. "
            "The pre-intervention period demonstrates model fit, while the post-intervention period displays "
            "the counterfactual prediction (what would have happened without intervention) compared to actual outcomes.",
            style={"color": "#64748b", "marginBottom": "1rem", "lineHeight": "1.6"}
        ),
        chart,
    ])
    
    # Add posterior distribution accordion if available
    posterior_mean_effect = experiment["results"].get("posterior_mean_effect", [])
    posterior_cumulative_effect = experiment["results"].get("posterior_cumulative_effect", [])
    posterior_accordion = _build_posterior_accordion(posterior_mean_effect, posterior_cumulative_effect)
    if posterior_accordion:
        components.append(posterior_accordion)
    
    # Add sensitivity analysis accordion if available
    sensitivity_analysis = experiment["results"].get("sensitivity_analysis")
    sensitivity_accordion = _build_sensitivity_accordion(sensitivity_analysis)
    if sensitivity_accordion:
        components.append(sensitivity_accordion)
    
    # Add AI-generated summary section if available
    ai_summary = experiment["results"].get("ai_summary")
    if ai_summary:
        components.append(
            html.Div(
                [
                    html.Div("AI-Generated Summary", className="ai-summary-title"),
                    dcc.Markdown(f"{ai_summary}", className="ai-summary-text"),
                    html.Div(
                        "⚠️ Please verify the interpretation below",
                        className="ai-summary-disclaimer"
                    ),
                ],
                className="ai-summary-card",
            )
        )
    
    return components


def _format_date(value: Optional[str]) -> str:
    if not value:
        return "Not set"
    try:
        parsed = datetime.fromisoformat(value).date()
        return f"{parsed:%b %d, %Y}"
    except ValueError:
        return value


def _format_model_name(model_string: Optional[str]) -> str:
    """Convert model string like 'cp.SyntheticControl' to 'Synthetic Control'."""
    if not model_string:
        return "N/A"
    
    # Remove 'cp.' prefix if present
    if model_string.startswith("cp."):
        model_string = model_string[3:]
    
    # Convert CamelCase to spaced words
    # SyntheticControl -> Synthetic Control
    # InterruptedTimeSeries -> Interrupted Time Series
    import re
    formatted = re.sub(r'([A-Z])', r' \1', model_string).strip()
    return formatted


app = Dash(__name__, server=server, url_base_pathname="/dash/")
app.title = "Experimentation Console"
app.config.suppress_callback_exceptions = True

# Configure timeout for experiment processing (default 3 minutes)
EXPERIMENT_TIMEOUT_SECONDS = int(os.getenv("EXPERIMENT_TIMEOUT_SECONDS", "180"))
logger.info(f"Experiment timeout configured to {EXPERIMENT_TIMEOUT_SECONDS} seconds")

# Global error handler for callbacks
def handle_callback_error(error):
    """Global error handler for unhandled callback exceptions."""
    logger.error(f"Unhandled callback error: {error}")
    logger.error(traceback.format_exc())
    return html.Div(
        [
            html.Span("⚠", className="error-notification-icon"),
            html.Div(f"An unexpected error occurred: {str(error)}", className="error-notification-message"),
        ],
        className="error-notification",
    )

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                font-family: "Inter", sans-serif;
                background-color: #f9fafb;
                color: #1f2933;
            }
            .app-shell {
                display: flex;
                min-height: 100vh;
            }
            .sidebar {
                width: 280px;
                max-width: 360px;
                background: #ffffff;
                border-right: 1px solid #e4e7eb;
                padding: 1.5rem 1rem;
                transition: width 0.3s ease;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                gap: 1rem;
            }
            .sidebar.expanded {
                width: 320px;
            }
            .sidebar.collapsed {
                width: 220px;
                padding: 1.5rem 0.75rem;
            }
            .sidebar-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.25rem;
            }
            .sidebar.collapsed .sidebar-title {
                font-size: 0.9rem;
            }
            .sidebar.collapsed .experiment-card-description,
            .sidebar.collapsed .experiment-card-meta {
                display: none;
            }
            .sidebar-title {
                font-size: 1rem;
                font-weight: 600;
            }
            .expand-button {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                border: 1px solid #d4dbe5;
                background: #f8fafc;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
            }
            .experiment-list {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                flex: 1;
                overflow-y: auto;
            }
            .experiment-card-container {
                position: relative;
                display: flex;
                width: 100%;
            }
            .experiment-card {
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
                padding: 0.75rem 0.9rem;
                border-radius: 12px;
                border: 1px solid #e5e9f2;
                background: #f8fafc;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                text-align: left;
                width: 100%;
                font-family: inherit;
                outline: none;
            }
            .experiment-card-container:has(.experiment-card.active) .experiment-card,
            .experiment-card.active {
                border-color: #2563eb;
                background: #eff6ff;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
                transform: translateX(4px);
            }
            .experiment-card-container:hover .experiment-card {
                border-color: #c0c9d9;
                transform: translateX(4px);
            }
            .experiment-card-title {
                font-weight: 600;
                font-size: 0.95rem;
                color: #1f2937;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .experiment-card-description {
                font-size: 0.8rem;
                color: #4b5563;
            }
            .experiment-card-meta {
                font-size: 0.8rem;
                color: #6b7280;
                display: flex;
                align-items: center;
                gap: 0.35rem;
                flex-wrap: nowrap;
            }
            .experiment-card-meta .dot {
                width: 6px;
                height: 6px;
                background: #9ca3af;
                border-radius: 50%;
                display: inline-block;
            }
            .main {
                flex: 1;
                padding: 2rem 3rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 2rem;
            }
            .top-bar {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 2rem;
            }
            .mode-toggle {
                display: flex;
                gap: 0.5rem;
                background: #f1f5f9;
                padding: 0.25rem;
                border-radius: 10px;
            }
            .mode-button {
                padding: 0.5rem 1.25rem;
                border: none;
                background: transparent;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 0.9rem;
                color: #64748b;
                transition: all 0.2s ease;
            }
            .mode-button:hover {
                color: #1f2937;
            }
            .mode-button.active {
                background: white;
                color: #1f2937;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .top-title {
                font-size: 1.5rem;
                font-weight: 600;
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 1rem;
                text-align: center;
            }
            .card {
                background: transparent;
                border-radius: 16px;
                padding: 1.5rem;
                border: none;
                box-shadow: none;
                max-width: 780px;
                margin: 0 auto;
                width: 100%;
            }
            .composer-wrapper {
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.75rem;
                margin: 0 auto 1.5rem;
                width: 100%;
                max-width: 720px;
            }
            .message-composer {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: space-between;
                width: 100%;
                padding: 0.35rem 0.5rem;
                border-radius: 999px;
                border: 1px solid #d6deeb;
                background: white;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
            }
            .composer-icon {
                width: 42px;
                height: 42px;
                border-radius: 999px;
                border: none;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.35rem;
                cursor: pointer;
                color: #1f2937;
                transition: background 0.2s ease;
            }
            .composer-icon:hover {
                background: rgba(37, 99, 235, 0.08);
            }
            .composer-icon.send {
                font-size: 1.2rem;
                color: #2563eb;
            }
            .composer-input {
                flex: 1;
                border: none;
                resize: none;
                padding: 0 0.25rem;
                font-size: 0.95rem;
                min-height: 52px;
                max-height: 120px;
                background: transparent;
            }
            .composer-input:focus {
                outline: none;
            }
            .attachment-popover {
                position: absolute;
                left: 50%;
                transform: translate(-50%, 0);
                width: min(560px, 85vw);
                background: rgba(255, 255, 255, 0.92);
                border-radius: 18px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
                border: 1px solid #e0e7ff;
                padding: 1.25rem;
                display: none;
                z-index: 50;
            }
            .attachment-popover::before {
                content: "";
                position: absolute;
                width: 18px;
                height: 18px;
                border-left: 1px solid #e0e7ff;
                border-bottom: 1px solid #e0e7ff;
                background: rgba(255, 255, 255, 0.92);
            }
            .attachment-popover.attachment-top {
                top: -10px;
                transform: translate(-50%, -100%);
            }
            .attachment-popover.attachment-top::before {
                bottom: -10px;
                left: 50%;
                transform: translateX(-50%) rotate(45deg);
            }
            .attachment-popover.attachment-bottom {
                top: calc(100% + 16px);
                transform: translate(-50%, 0);
            }
            .attachment-popover.attachment-bottom::before {
                top: -10px;
                left: 50%;
                transform: translateX(-50%) rotate(225deg);
            }
            .attachment-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            .attachment-column {
                display: flex;
                flex-direction: column;
                gap: 0.65rem;
            }
            .attachment-label {
                font-size: 0.8rem;
                font-weight: 600;
                color: #4b5563;
            }
            .composer-chip-row {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                justify-content: center;
                min-height: 28px;
            }
            .composer-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                background: rgba(37, 99, 235, 0.15);
                color: #1d4ed8;
                font-size: 0.75rem;
                font-weight: 600;
                border: none;
            }
            .upload-feedback-container {
                display: flex;
                justify-content: center;
                margin-bottom: 0.5rem;
            }
            .upload-feedback {
                display: inline-flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0.75rem;
                border-radius: 12px;
                background: rgba(37, 99, 235, 0.12);
                color: #1e3a8a;
                font-size: 0.85rem;
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.15);
            }
            .upload-status {
                font-weight: 600;
            }
            .detail-card {
                margin-bottom: 1.5rem;
            }
            .message-card {
                background: #f8fafc;
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid #e2e8f0;
                margin-top: 0.75rem;
                margin-bottom: 1.25rem;
            }
            .message-title {
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .meta-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 1rem;
            }
            .meta-label {
                font-size: 0.875rem;
                color: #60708c;
            }
            .meta-value {
                font-weight: 600;
                margin-top: 0.25rem;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            .metric-card {
                background: #f8fafc;
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid #e2e8f0;
            }
            .metric-label {
                font-size: 0.875rem;
                color: #475569;
            }
            .metric-value {
                margin-top: 0.35rem;
                font-size: 1.5rem;
                font-weight: 600;
            }
            .metric-subtitle {
                margin-top: 0.25rem;
                font-size: 0.75rem;
                color: #64748b;
                font-weight: 400;
            }
            .section-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            .section-body {
                background: #f8fafc;
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid #e2e8f0;
            }
            .upload-box {
                border: 1px dashed #94a3b8;
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                background: white;
            }
            .logout-container {
                margin-top: auto;
                display: flex;
                justify-content: center;
                padding-top: 1rem;
            }
            .logout-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 1.35rem;
                color: #6b7280;
                text-decoration: none;
                padding: 0.5rem 0.75rem;
                border-radius: 10px;
                transition: color 0.2s ease, background 0.2s ease;
            }
            .logout-button:hover {
                color: #ef4444;
                background: rgba(239, 68, 68, 0.12);
            }
            .error-notification {
                background: #fee;
                border: 1px solid #fcc;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                color: #c33;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .error-notification-icon {
                font-size: 1.5rem;
            }
            .error-notification-message {
                flex: 1;
            }
            .ai-summary-card {
                background: #fffbeb;
                border: 1px solid #fbbf24;
                border-left: 5px solid #f59e0b;
                border-radius: 12px;
                padding: 1.5rem;
                margin-top: 2rem;
                box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
            }
            .ai-summary-title {
                font-size: 1.15rem;
                font-weight: 600;
                color: #92400e;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .ai-summary-title::before {
                content: "💡";
                font-size: 1.25rem;
            }
            .ai-summary-disclaimer {
                background: #fef3c7;
                border-left: 3px solid #f59e0b;
                padding: 0.65rem 0.85rem;
                margin-bottom: 1rem;
                font-size: 0.8rem;
                color: #78350f;
                border-radius: 6px;
                font-weight: 500;
            }
            .ai-summary-text {
                color: #1f2937;
                line-height: 1.8;
                font-size: 0.95rem;
                text-align: justify;
                padding: 0.5rem 0;
            }
            .experiment-card.loading {
                opacity: 0.7;
                cursor: not-allowed;
                pointer-events: none;
                background: #f1f5f9;
            }
            .loading-spinner {
                display: inline-block;
                width: 14px;
                height: 14px;
                border: 2px solid #e5e9f2;
                border-top: 2px solid #2563eb;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 0.5rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .experiment-card-menu-button {
                position: absolute;
                bottom: 0.5rem;
                right: 0.5rem;
                width: 24px;
                height: 24px;
                border: none;
                background: transparent;
                color: #9ca3af;
                font-size: 1rem;
                cursor: pointer;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s ease, color 0.2s ease;
                z-index: 100;
                padding: 0;
                line-height: 1;
            }
            .experiment-card-menu-button:hover {
                background: rgba(239, 68, 68, 0.12);
                color: #ef4444;
            }
            .delete-confirmation-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.4);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                pointer-events: auto;
            }
            .delete-confirmation-modal {
                background: white;
                border-radius: 16px;
                padding: 2rem;
                max-width: 420px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                z-index: 1001;
                pointer-events: auto;
            }
            .delete-confirmation-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
                color: #1f2937;
            }
            .delete-confirmation-message {
                color: #64748b;
                margin-bottom: 1.5rem;
                line-height: 1.6;
            }
            .delete-confirmation-buttons {
                display: flex;
                gap: 0.75rem;
                justify-content: flex-end;
            }
            .delete-confirmation-button {
                padding: 0.5rem 1rem;
                border-radius: 8px;
                border: 1px solid #d1d5db;
                background: white;
                font-size: 0.95rem;
                cursor: pointer;
                transition: all 0.2s ease;
                position: relative;
                z-index: 1002;
                pointer-events: auto;
            }
            .delete-confirmation-button.cancel {
                color: #4b5563;
            }
            .delete-confirmation-button.cancel:hover {
                background: #f3f4f6;
            }
            .delete-confirmation-button.confirm {
                background: #ef4444;
                color: white;
                border-color: #ef4444;
            }
            .delete-confirmation-button.confirm:hover {
                background: #dc2626;
            }
            .chat-toggle-button {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                border: none;
                background: #2563eb;
                color: white;
                font-size: 1.5rem;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
                transition: all 0.2s ease;
                z-index: 999;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .chat-toggle-button:hover {
                background: #1d4ed8;
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
            }
            .chat-panel {
                position: fixed;
                right: 0;
                top: 0;
                bottom: 0;
                width: 33.33%;
                background: white;
                box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
                transform: translateX(100%);
                transition: transform 0.3s ease;
                z-index: 1000;
                display: flex;
                flex-direction: column;
            }
            .chat-panel.open {
                transform: translateX(0);
            }
            .chat-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1rem 1.25rem;
                border-bottom: 1px solid #e5e7eb;
                background: #f9fafb;
            }
            .chat-header-title {
                font-size: 1.125rem;
                font-weight: 600;
                color: #1f2937;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .chat-close-button {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                border: none;
                background: transparent;
                color: #6b7280;
                font-size: 1.25rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .chat-close-button:hover {
                background: #f3f4f6;
                color: #1f2937;
            }
            .chat-messages-container {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .chat-message {
                display: flex;
                gap: 0.75rem;
                align-items: flex-start;
            }
            .chat-message.user {
                flex-direction: row-reverse;
            }
            .chat-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.125rem;
                flex-shrink: 0;
            }
            .chat-avatar.user {
                background: #dbeafe;
                color: #2563eb;
            }
            .chat-avatar.assistant {
                background: #f3f4f6;
                color: #6b7280;
            }
            .chat-bubble {
                max-width: 70%;
                padding: 0.75rem 1rem;
                border-radius: 12px;
                word-wrap: break-word;
                white-space: pre-wrap;
                line-height: 1.5;
            }
            .chat-bubble.user {
                background: #dbeafe;
                color: #1e3a8a;
                border: 1px solid #bfdbfe;
            }
            .chat-bubble.assistant {
                background: #f8f9fa;
                color: #1f2937;
                border: 1px solid #e5e7eb;
            }
            .chat-input-container {
                padding: 1rem;
                border-top: 1px solid #e5e7eb;
                background: #f9fafb;
            }
            .chat-input-wrapper {
                display: flex;
                gap: 0.5rem;
                align-items: flex-end;
            }
            .chat-input {
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 0.95rem;
                resize: none;
                min-height: 44px;
                max-height: 120px;
                font-family: inherit;
            }
            .chat-input:focus {
                outline: none;
                border-color: #2563eb;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }
            .chat-input:disabled {
                background: #f3f4f6;
                cursor: not-allowed;
            }
            .chat-send-button {
                width: 44px;
                height: 44px;
                border-radius: 8px;
                border: none;
                background: #2563eb;
                color: white;
                font-size: 1.25rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }
            .chat-send-button:hover:not(:disabled) {
                background: #1d4ed8;
            }
            .chat-send-button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }
            .validation-error {
                color: #ef4444;
                font-size: 0.75rem;
                margin-top: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.25rem;
            }
            .chat-empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                color: #9ca3af;
                text-align: center;
                padding: 2rem;
            }
            .chat-empty-state-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            .planning-messages-area {
                max-width: 720px;
                margin: 0 auto 2rem auto;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                min-height: 200px;
            }
            .planning-message {
                display: flex;
                gap: 0.75rem;
                align-items: flex-start;
            }
            .planning-message.user {
                flex-direction: row-reverse;
            }
            .planning-bubble {
                max-width: 70%;
                padding: 0.75rem 1rem;
                border-radius: 12px;
                word-wrap: break-word;
                white-space: pre-wrap;
                line-height: 1.5;
            }
            .planning-bubble.user {
                background: #dbeafe;
                color: #1e3a8a;
                border: 1px solid #bfdbfe;
            }
            .planning-bubble.assistant {
                background:rgb(243, 246, 250);
                color: #1f2937;
                border: 1px solid rgb(210, 213, 220);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""


def _initial_experiments_state() -> List[Dict[str, Any]]:
    """Load initial experiments state from storage for the current user."""
    if not has_request_context():
        logger.debug("No request context available, returning empty experiments list")
        return []

    user_key = session.get("user", "default")
    logger.info(f"Loading initial experiments state for user: {user_key}")
    
    # Always load from storage (storage is the source of truth)
    # DO NOT store in session - experiments data is too large for cookie storage
    experiments = load_experiments(user_key)
    logger.info(f"Loaded {len(experiments)} experiments from storage for user: {user_key}")
    
    logger.info(f"Returning {len(experiments)} experiments for initial state")
    return experiments


def _initial_planning_sessions_state() -> List[Dict[str, Any]]:
    """Load initial planning sessions state from storage for the current user."""
    if not has_request_context():
        logger.debug("No request context available, returning empty planning sessions list")
        return []

    user_key = session.get("user", "default")
    logger.info(f"Loading initial planning sessions state for user: {user_key}")
    
    # Always load from storage (storage is the source of truth)
    # DO NOT store in session - planning sessions data is too large for cookie storage
    sessions = load_planning_sessions(user_key)
    logger.info(f"Loaded {len(sessions)} planning sessions from storage for user: {user_key}")
    
    logger.info(f"Returning {len(sessions)} planning sessions for initial state")
    return sessions


def serve_layout():
    initial_experiments = _initial_experiments_state()
    initial_planning_sessions = _initial_planning_sessions_state()
    return dmc.MantineProvider(
        html.Div(
            [
                dcc.Store(id="experiments-store", data=initial_experiments),
                dcc.Store(id="sidebar-expanded", data=True),
                dcc.Store(id="selected-experiment", data="New Experiment"),
                dcc.Store(id="uploaded-filename", data=None),
                dcc.Store(id="error-message", data=None),
                dcc.Store(id="processing-trigger", data=None),
                dcc.Store(id="processing-metadata", data={}),
                dcc.Store(id="delete-confirmation-experiment", data=None),
                dcc.Store(id="delete-confirmation-visible", data=False),
                dcc.Store(id="chat-visible", data=False),
                dcc.Store(id="chat-messages", data=[]),
                dcc.Store(id="chat-validation-error", data=None),
                dcc.Store(id="chat-loading", data=False),
                # Planning mode stores
                dcc.Store(id="app-mode", data="experiments"),
                dcc.Store(id="planning-sessions-store", data=initial_planning_sessions),
                dcc.Store(id="selected-planning-session", data="New Planning"),
                dcc.Store(id="planning-messages", data=[]),
                dcc.Store(id="planning-validation-error", data=None),
                dcc.Interval(id="cleanup-interval", interval=30000, n_intervals=0),  # Check every 30 seconds
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Library", className="sidebar-title"),
                                        html.Button("⟷", id="sidebar-toggle", className="expand-button", n_clicks=0),
                                    ],
                                    className="sidebar-header",
                                ),
                                html.Div(id="experiment-list", className="experiment-list"),
                                html.Div(
                                    dcc.Link(
                                        "⏻",
                                        className="logout-button",
                                        title="Log out",
                                        href="/logout",
                                    ),
                                    className="logout-container",
                                ),
                            ],
                            id="sidebar",
                            className="sidebar expanded",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Button("Experiments", id="mode-experiments", className="mode-button active"),
                                                html.Button("Planning", id="mode-planning", className="mode-button"),
                                            ],
                                            className="mode-toggle",
                                        ),
                                    ],
                                    className="top-bar",
                                ),
                                html.Div(
                                    [
                                html.Div(id="error-notification", style={"display": "none"}),
                                html.Div(
                                    [
                                        html.H2("What are we testing today?", className="hero-title"),
                                        html.Div(
                                            [
                                                html.Div("🔬", style={"fontSize": "3rem", "marginBottom": "1rem"}),
                                                html.Div(
                                                    "Describe your experiment, upload your data, and let AI guide your causal inference analysis.",
                                                    style={"fontSize": "0.875rem", "color": "#94a3b8", "marginBottom": "2rem", "textAlign": "center", "maxWidth": "600px"}
                                                ),
                                            ],
                                            style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Button("＋", id="toggle-attachments", className="composer-icon"),
                                                        dcc.Textarea(
                                                            id="form-message",
                                                            placeholder="Tell me about the intervention you ran…",
                                                            className="composer-input",
                                                            value="",
                                                        ),
                                                        html.Button("➤", id="composer-send", className="composer-icon send"),
                                                    ],
                                                    id="message-composer",
                                                    className="message-composer",
                                                ),
                                                html.Div(id="composer-chips", className="composer-chip-row"),
                                                html.Div(id="upload-feedback", className="upload-feedback-container"),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div("Dataset", className="attachment-label"),
                                                                dcc.Upload(
                                                                    id="upload-data",
                                                                    children=html.Div(
                                                                        [
                                                                            "Drag and drop or ",
                                                                            html.Span("browse", style={"color": "#2563eb", "fontWeight": "600"}),
                                                                        ],
                                                                        className="upload-box",
                                                                    ),
                                                                    multiple=False,
                                                                ),
                                                            ],
                                                            className="attachment-column",
                                                        ),
                                                    ],
                                                    id="attachment-area",
                                                    className="attachment-popover attachment-top",
                                                    style={"display": "none"},
                                                ),
                                            ],
                                            className="composer-wrapper",
                                            id="composer-wrapper",
                                        ),
                                    ],
                                    id="new-experiment-view",
                                    className="card",
                                ),
                                html.Div(id="experiment-detail-view", className="card", style={"display": "none"}),
                                # Planning view
                                html.Div(
                                    [
                                        html.H2("What would you like to plan?", className="hero-title"),
                                        html.Div(id="planning-messages-container", className="planning-messages-area"),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        dcc.Textarea(
                                                            id="planning-input",
                                                            placeholder="Your question...",
                                                            className="composer-input",
                                                            value="",
                                                        ),
                                                        html.Button("➤", id="planning-send", className="composer-icon send"),
                                                    ],
                                                    id="planning-composer",
                                                    className="message-composer",
                                                ),
                                                html.Div(id="planning-validation-error-display", style={"display": "none"}),
                                            ],
                                            className="composer-wrapper",
                                            id="planning-composer-wrapper",
                                        ),
                                    ],
                                    id="planning-view",
                                    className="card",
                                    style={"display": "none"},
                                ),
                                    ],
                                    className="content-area",
                                ),
                            ],
                            className="main",
                            id="main-container",
                        ),
                    ],
                    className="app-shell",
                ),
                html.Div(id="delete-confirmation-modal", style={"display": "none"}),
                # Chat toggle button
                html.Button(
                    "💬",
                    id="chat-toggle-button",
                    className="chat-toggle-button",
                    style={"display": "none"},
                    n_clicks=0,
                ),
                # Chat panel
                html.Div(
                    id="chat-panel",
                    className="chat-panel",
                    children=[
                        # Header
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("💬", style={"fontSize": "1.25rem"}),
                                        html.Span("Experiment Chat"),
                                    ],
                                    className="chat-header-title",
                                ),
                                html.Button(
                                    "✕",
                                    id="chat-close-button",
                                    className="chat-close-button",
                                    n_clicks=0,
                                ),
                            ],
                            className="chat-header",
                        ),
                        # Messages area
                        html.Div(
                            id="chat-messages-container",
                            className="chat-messages-container",
                            children=[
                                html.Div(
                                    [
                                        html.Div("💬", className="chat-empty-state-icon"),
                                        html.Div(
                                            "Ask me anything about this experiment!",
                                            style={"fontSize": "1.125rem", "fontWeight": "600", "marginBottom": "0.5rem"}
                                        ),
                                        html.Div(
                                            "I can help explain the results, methodology, and statistical concepts.",
                                            style={"fontSize": "0.875rem"}
                                        ),
                                    ],
                                    className="chat-empty-state",
                                ),
                            ],
                        ),
                        # Input area
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Textarea(
                                            id="chat-input",
                                            className="chat-input",
                                            placeholder="Ask a question about this experiment...",
                                            value="",
                                        ),
                                        html.Button(
                                            "➤",
                                            id="chat-send-button",
                                            className="chat-send-button",
                                            n_clicks=0,
                                        ),
                                    ],
                                    className="chat-input-wrapper",
                                ),
                                html.Div(id="chat-validation-error-display", style={"display": "none"}),
                            ],
                            className="chat-input-container",
                        ),
                    ],
                ),
            ],
        )
    )


app.layout = serve_layout


@app.callback(
    Output("experiment-list", "children"),
    Output("sidebar-expanded", "data"),
    Output("sidebar", "className"),
    Input("experiments-store", "data"),
    Input("selected-experiment", "data"),
    Input("sidebar-toggle", "n_clicks"),
    Input("planning-sessions-store", "data"),
    Input("selected-planning-session", "data"),
    Input("app-mode", "data"),
    State("sidebar-expanded", "data"),
)
def update_sidebar(
    experiments: Optional[List[Dict[str, Any]]],
    selected: Optional[str],
    toggle_clicks: Optional[int],
    planning_sessions: Optional[List[Dict[str, Any]]],
    selected_planning: Optional[str],
    mode: Optional[str],
    expanded_state: Optional[bool],
):
    mode = mode or "experiments"
    experiments = experiments or []
    planning_sessions = planning_sessions or []
    
    # Determine which mode we're in and which selection to use
    if mode == "planning":
        selected_value = selected_planning or "New Planning"
        cards = [
            html.Div(
                html.Button(
                    [
                        html.Div("New Planning", className="experiment-card-title"),
                        html.Div("Start a new planning session", className="experiment-card-description"),
                    ],
                    className=f"experiment-card{' active' if selected_value == 'New Planning' else ''}",
                    id={"type": "planning-card", "value": "New Planning"},
                    n_clicks=0,
                ),
                className="experiment-card-container",
            )
        ]
        
        # Add planning session cards
        for session in planning_sessions:
            # Get first user message as preview
            messages = session.get("messages", [])
            preview = "No messages yet"
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    preview = content[:60] + ("…" if len(content) > 60 else "")
                    break
            
            timestamp = session.get("updated_at") or session.get("created_at", "")
            
            cards.append(
                html.Div(
                    [
                        html.Button(
                            [
                                html.Div(session["name"], className="experiment-card-title"),
                                html.Div(preview, className="experiment-card-description"),
                                html.Div(
                                    _format_date(timestamp) if timestamp else "",
                                    className="experiment-card-meta",
                                ),
                            ],
                            className=f"experiment-card{' active' if selected_value == session['name'] else ''}",
                            id={"type": "planning-card", "value": session["name"]},
                            n_clicks=0,
                        ),
                        html.Button(
                            "⋮",
                            id={"type": "planning-menu-button", "session": session["name"]},
                            className="experiment-card-menu-button",
                            n_clicks=0,
                        ),
                    ],
                    className="experiment-card-container",
                )
            )
    else:
        # Experiments mode (original behavior)
        selected_value = selected or "New Experiment"
        cards = [
            html.Div(
                html.Button(
                    [
                        html.Div("New Experiment", className="experiment-card-title"),
                        html.Div("Start a fresh request", className="experiment-card-description"),
                    ],
                    className=f"experiment-card{' active' if selected_value == 'New Experiment' else ''}",
                    id={"type": "experiment-card", "value": "New Experiment"},
                    n_clicks=0,
                ),
                className="experiment-card-container",
            )
        ]

        # Add experiment cards (only in experiments mode)
        for exp in experiments:
            subtitle = exp.get("message", "").strip() or "Untitled experiment request"
            is_loading = exp.get("status") == "loading"
            
            if is_loading:
                # Render loading experiment with spinner
                cards.append(
                    html.Div(
                        html.Button(
                            [
                                html.Div(
                                    [
                                        html.Span(className="loading-spinner"),
                                        html.Span(exp["name"]),
                                    ],
                                    className="experiment-card-title",
                                    style={"display": "flex", "alignItems": "center"}
                                ),
                                html.Div("Processing...", className="experiment-card-description"),
                            ],
                            className=f"experiment-card loading{' active' if selected_value == exp['name'] else ''}",
                            id={"type": "experiment-card", "value": exp["name"]},
                            n_clicks=0,
                        ),
                        className="experiment-card-container",
                    )
                )
            else:
                # Render completed experiment normally - menu button is sibling, not nested
                cards.append(
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Div(exp["name"], className="experiment-card-title"),
                                    html.Div(subtitle[:60] + ("…" if len(subtitle) > 60 else ""), className="experiment-card-description"),
                                    html.Div(
                                        [
                                            html.Div(_format_date(exp.get("start_date")), className="meta-value"),
                                            html.Span(className="dot"),
                                            html.Div(_format_date(exp.get("end_date")), className="meta-value"),
                                        ],
                                        className="experiment-card-meta",
                                    ),
                                ],
                                className=f"experiment-card{' active' if selected_value == exp['name'] else ''}",
                                id={"type": "experiment-card", "value": exp["name"]},
                                n_clicks=0,
                            ),
                            html.Button(
                                "⋮",
                                id={"type": "experiment-menu-button", "experiment": exp["name"]},
                                className="experiment-card-menu-button",
                                n_clicks=0,
                            ),
                        ],
                        className="experiment-card-container",
                    )
                )

    expanded = True if expanded_state is None else bool(expanded_state)
    ctx = callback_context.triggered_id
    if ctx == "sidebar-toggle":
        expanded = not expanded

    sidebar_class = "sidebar expanded" if expanded else "sidebar collapsed"

    return cards, expanded, sidebar_class


@app.callback(
    Output("selected-experiment", "data", allow_duplicate=True),
    Output("error-notification", "children", allow_duplicate=True),
    Output("error-notification", "style", allow_duplicate=True),
    Input({"type": "experiment-card", "value": ALL}, "n_clicks"),
    State({"type": "experiment-card", "value": ALL}, "id"),
    State("selected-experiment", "data"),
    State("experiments-store", "data"),
    prevent_initial_call=True,
)
def select_experiment(cards_clicks, card_ids, current_selection, experiments):
    triggered = callback_context.triggered_id

    if not triggered:
        return current_selection, None, {"display": "none"}

    if isinstance(triggered, dict):
        experiment_name = triggered.get("value", current_selection)
        
        # Check if the selected experiment is in loading state
        experiments = experiments or []
        for exp in experiments:
            if exp.get("name") == experiment_name and exp.get("status") == "loading":
                # Show error message that experiment is still processing
                error_msg = html.Div(
                    [
                        html.Span("⚠", className="error-notification-icon"),
                        html.Div("This experiment is still processing. Please wait...", className="error-notification-message"),
                    ],
                    className="error-notification",
                )
                return current_selection, error_msg, {"display": "block"}
        
        # Not loading, allow selection
        return experiment_name, None, {"display": "none"}

    return current_selection, None, {"display": "none"}


@app.callback(
    Output("new-experiment-view", "style"),
    Output("experiment-detail-view", "style"),
    Output("experiment-detail-view", "children"),
    Output("planning-view", "style"),
    Input("selected-experiment", "data"),
    Input("experiments-store", "data"),
    Input("selected-planning-session", "data"),
    Input("planning-sessions-store", "data"),
    Input("app-mode", "data"),
)
def update_main_view(
    selection: Optional[str],
    experiments: Optional[List[Dict[str, Any]]],
    planning_selection: Optional[str],
    planning_sessions: Optional[List[Dict[str, Any]]],
    mode: Optional[str],
):
    mode = mode or "experiments"
    experiments = experiments or []
    planning_sessions = planning_sessions or []
    
    # Planning mode
    if mode == "planning":
        selected_value = planning_selection or "New Planning"
        
        # Always show planning view in planning mode
        return (
            {"display": "none"},  # Hide new-experiment-view
            {"display": "none"},  # Hide experiment-detail-view
            [],                   # Empty experiment-detail children
            {"display": "block"}, # Show planning-view
        )
    
    # Experiments mode (original behavior)
    selected_value = selection or "New Experiment"

    if selected_value != "New Experiment":
        match = next((exp for exp in experiments if exp["name"] == selection), None)
        if match:
            # Check if experiment is still loading
            if match.get("status") == "loading":
                # Show loading view
                loading_view = [
                    html.Div(
                        [
                            html.Div(className="loading-spinner", style={"width": "48px", "height": "48px", "border-width": "4px"}),
                            html.H2("Processing experiment...", style={"marginTop": "2rem", "color": "#64748b"}),
                            html.P(
                                "Your experiment is being analyzed. This may take a few moments.",
                                style={"color": "#94a3b8", "marginTop": "0.5rem"}
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "minHeight": "400px"}
                    )
                ]
                return (
                    {"display": "none"},  # Hide new-experiment-view
                    {"display": "block"}, # Show experiment-detail-view
                    loading_view,
                    {"display": "none"},  # Hide planning-view
                )
            
            # Show completed experiment details
            return (
                {"display": "none"},  # Hide new-experiment-view
                {"display": "block"}, # Show experiment-detail-view
                _render_experiment_detail(match),
                {"display": "none"},  # Hide planning-view
            )
    return (
        {"display": "block"},  # Show new-experiment-view
        {"display": "none"},   # Hide experiment-detail-view
        [],                    # Empty experiment-detail children
        {"display": "none"},   # Hide planning-view
    )


@app.callback(
    Output("experiments-store", "data", allow_duplicate=True),
    Output("processing-trigger", "data"),
    Output("selected-experiment", "data", allow_duplicate=True),
    Output("composer-chips", "children"),
    Output("upload-feedback", "children", allow_duplicate=True),
    Output("attachment-area", "style", allow_duplicate=True),
    Output("attachment-area", "className", allow_duplicate=True),
    Output("form-message", "value", allow_duplicate=True),
    Output("upload-data", "contents", allow_duplicate=True),
    Output("upload-data", "filename", allow_duplicate=True),
    Output("error-notification", "children", allow_duplicate=True),
    Output("error-notification", "style", allow_duplicate=True),
    Output("processing-metadata", "data", allow_duplicate=True),
    Input("composer-send", "n_clicks"),
    State("experiments-store", "data"),
    State("form-message", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("attachment-area", "className"),
    State("processing-metadata", "data"),
    prevent_initial_call=True,
)
def add_loading_experiment(
    send_clicks: Optional[int],
    experiments: Optional[List[Dict[str, Any]]],
    message: str,
    upload_contents: Optional[str],
    upload_filename: Optional[str],
    current_class: str,
    processing_metadata: Optional[Dict[str, Any]],
):
    logger.info("=== add_loading_experiment called ===")
    triggered = callback_context.triggered_id
    logger.info(f"Triggered by: {triggered}")
    
    if triggered != "composer-send":
        logger.info("Not triggered by composer-send, preventing update")
        raise PreventUpdate

    if not send_clicks:
        logger.warning("No send_clicks, preventing update")
        raise PreventUpdate

    experiments = experiments or []
    logger.info(f"Current experiment count: {len(experiments)}")
    
    try:
        # Basic validation
        if not message.strip():
            logger.error("Empty message received")
            raise ExperimentFailure("Experiment request cannot be empty.")

        if not upload_contents or not upload_filename:
            logger.error("No dataset uploaded")
            raise ExperimentFailure("Please attach a dataset before submitting.")

        logger.info(f"Message: {message[:100]}...")
        logger.info(f"Upload filename: {upload_filename}")

        # Generate name for the loading experiment
        name = _generate_experiment_name(len(experiments) + 1)
        logger.info(f"Generated experiment name: {name}")
        
        # Create loading experiment entry
        loading_experiment = {
            "name": name,
            "message": message.strip() or "Untitled experiment request",
            "status": "loading",
        }

        # Add loading experiment to the list
        updated_experiments = experiments + [loading_experiment]
        
        # Store form data for the processing callback
        processing_data = {
            "experiment_name": name,
            "message": message,
            "upload_contents": upload_contents,
            "upload_filename": upload_filename,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"Added loading experiment: {name}")
        logger.info(f"Updated experiments count: {len(updated_experiments)}")
        logger.info(f"Triggering process_experiment with data for: {name}")
        
        chips = build_composer_chips(upload_filename)
        upload_feedback = build_upload_message(upload_filename)
        
        # Update processing metadata
        processing_metadata = processing_metadata or {}
        processing_metadata[name] = {
            "start_time": datetime.now().isoformat(),
            "status": "processing",
        }

        logger.info("=== add_loading_experiment returning ===")
        return (
            updated_experiments,
            processing_data,  # Trigger processing callback
            name,  # Don't select yet, let the processing callback do it
            chips,
            upload_feedback,
            {"display": "none"},
            "attachment-popover attachment-top",
            "",  # Clear message
            None,  # Keep upload contents for processing callback
            None,  # Keep upload filename for processing callback
            None,  # Clear error notification
            {"display": "none"},
            processing_metadata,  # Update processing metadata
        )
    except ExperimentFailure as exc:
        logger.error(f"ExperimentFailure: {exc}")
        error_msg = html.Div(
            [
                html.Span("⚠", className="error-notification-icon"),
                html.Div(str(exc), className="error-notification-message"),
            ],
            className="error-notification",
        )
        from dash import no_update
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            error_msg,
            {"display": "block"},
            no_update,  # processing-metadata
        )


@app.callback(
    Output("experiments-store", "data", allow_duplicate=True),
    Output("selected-experiment", "data", allow_duplicate=True),
    Output("upload-data", "contents"),
    Output("upload-data", "filename"),
    Output("error-notification", "children", allow_duplicate=True),
    Output("error-notification", "style", allow_duplicate=True),
    Output("processing-metadata", "data", allow_duplicate=True),
    Input("processing-trigger", "data"),
    State("experiments-store", "data"),
    State("processing-metadata", "data"),
    prevent_initial_call=True,
)
def process_experiment(
    processing_data: Optional[Dict[str, Any]],
    experiments: Optional[List[Dict[str, Any]]],
    processing_metadata: Optional[Dict[str, Any]],
):
    """Process the experiment and replace the loading entry with results.
    
    This callback has guaranteed cleanup: loading experiments are ALWAYS removed,
    either by being replaced with results (success) or deleted (failure).
    All return paths update processing_metadata to track completion.
    """
    logger.info("=== process_experiment called ===")
    logger.info(f"Processing data received: {processing_data is not None}")
    logger.info(f"Experiments count: {len(experiments) if experiments else 0}")
    
    if not processing_data:
        logger.info("No processing data, preventing update")
        raise PreventUpdate
    
    user_key = session.get("user", "default")
    logger.info(f"User key: {user_key}")
    
    experiments = experiments or []
    processing_metadata = processing_metadata or {}
    experiment_name = processing_data.get("experiment_name")
    message = processing_data.get("message")
    upload_contents = processing_data.get("upload_contents")
    upload_filename = processing_data.get("upload_filename")
    
    logger.info(f"Processing experiment: {experiment_name}")
    logger.info(f"Message length: {len(message) if message else 0}")
    logger.info(f"Upload filename: {upload_filename}")
    logger.info(f"Has upload contents: {upload_contents is not None}")
    
    # Variables for cleanup tracking in finally block
    updated_experiments = experiments
    final_selection = "New Experiment"
    error_msg = None
    error_style = {"display": "none"}

    try:
        # Parse the uploaded dataset
        df = parse_upload(upload_contents, upload_filename)
        if df is None:
            logger.error("Failed to parse upload")
            raise ExperimentFailure("Failed to parse the uploaded dataset.")
        
        logger.info(f"Parsed dataframe with shape: {df.shape}, columns: {df.columns.tolist()}")

        # Get OpenAI client
        client = current_app.config.get("OPENAI_CLIENT")
        if client is None:
            logger.error("OpenAI client not configured")
            raise ExperimentFailure("OpenAI client is not configured.")

        logger.info("Creating LLM extractor and running experiment pipeline...")
        extractor = LLMExtractor(client)
        
        try:
            results_payload = run_experiment_pipeline(extractor, message, df)
            logger.info(f"Experiment pipeline completed. Result keys: {results_payload.keys()}")
        except ExtractionError as exc:
            logger.error(f"ExtractionError during pipeline: {exc}")
            logger.error(traceback.format_exc())
            raise ExperimentFailure(str(exc)) from exc
        except Exception as exc:
            logger.error(f"Unexpected error during pipeline: {exc}")
            logger.error(traceback.format_exc())
            raise ExperimentFailure(f"Experiment failed: {str(exc)}") from exc

        # Create completed experiment data
        experiment_data = {
            "name": experiment_name,
            "message": message.strip() or "Untitled experiment request",
            "start_date": results_payload.get("intervention_start_date"),
            "end_date": results_payload.get("intervention_end_date"),
            "dataset_name": upload_filename,
            "results": results_payload,
            "status": "complete",
        }

        # Replace loading experiment with completed one
        updated_experiments = [
            experiment_data if exp.get("name") == experiment_name else exp
            for exp in experiments
        ]
        
        final_selection = experiment_name  # Select the completed experiment
        error_msg = None
        error_style = {"display": "none"}
        
        logger.info(f"Experiment completed successfully! Total experiments: {len(updated_experiments)}")
        logger.info(f"Switching to experiment: {experiment_name}")
    except ExperimentFailure as exc:
        logger.error(f"=== ExperimentFailure in process_experiment ===")
        logger.error(f"Error: {exc}")
        logger.error(traceback.format_exc())
        
        # Remove the loading experiment from the list
        updated_experiments = [
            exp for exp in experiments
            if exp.get("name") != experiment_name
        ]
        logger.info(f"Removed loading experiment, remaining: {len(updated_experiments)}")
        
        final_selection = "New Experiment"
        error_msg = html.Div(
            [
                html.Span("⚠", className="error-notification-icon"),
                html.Div(str(exc), className="error-notification-message"),
            ],
            className="error-notification",
        )
        error_style = {"display": "block"}
        
    except Exception as exc:
        logger.error(f"=== Unexpected exception in process_experiment ===")
        logger.error(f"Exception type: {type(exc).__name__}")
        logger.error(f"Error: {exc}")
        logger.error(traceback.format_exc())
        
        # Remove the loading experiment from the list
        updated_experiments = [
            exp for exp in experiments
            if exp.get("name") != experiment_name
        ]
        logger.info(f"Removed loading experiment, remaining: {len(updated_experiments)}")
        
        final_selection = "New Experiment"
        error_msg = html.Div(
            [
                html.Span("⚠", className="error-notification-icon"),
                html.Div(f"An unexpected error occurred: {str(exc)}", className="error-notification-message"),
            ],
            className="error-notification",
        )
        error_style = {"display": "block"}
    
    finally:
        # GUARANTEED CLEANUP: Always save state and clear processing metadata
        # This ensures the UI never gets stuck in loading state
        
        # Save to file storage (DO NOT use session - too large for cookies)
        save_experiments(user_key, updated_experiments)
        
        # Clear processing metadata for this experiment
        if experiment_name in processing_metadata:
            processing_metadata[experiment_name] = {
                "start_time": processing_metadata[experiment_name].get("start_time"),
                "status": "complete",
                "end_time": datetime.now().isoformat(),
            }
        
        logger.info(f"=== process_experiment cleanup complete ===")
        logger.info(f"Final experiments count: {len(updated_experiments)}")
        logger.info(f"Final selection: {final_selection}")
        logger.info(f"Has error: {error_msg is not None}")
        
        return (
            updated_experiments,
            final_selection,
            None,  # Clear upload contents
            None,  # Clear upload filename
            error_msg,
            error_style,
            processing_metadata,
        )


@app.callback(
    Output("experiments-store", "data", allow_duplicate=True),
    Output("processing-metadata", "data", allow_duplicate=True),
    Input("cleanup-interval", "n_intervals"),
    State("experiments-store", "data"),
    State("processing-metadata", "data"),
    prevent_initial_call=True,
)
def cleanup_stuck_experiments(
    n_intervals: int,
    experiments: Optional[List[Dict[str, Any]]],
    processing_metadata: Optional[Dict[str, Any]],
):
    """Monitor and cleanup experiments stuck in loading state.
    
    This is a safety net that runs every 30 seconds to check for experiments
    that have been processing for longer than EXPERIMENT_TIMEOUT_SECONDS.
    If found, they are removed to prevent permanent loading state.
    """
    if not experiments or not processing_metadata:
        raise PreventUpdate
    
    experiments = experiments or []
    processing_metadata = processing_metadata or {}
    current_time = datetime.now()
    
    # Find experiments stuck in loading state
    stuck_experiments = []
    for exp in experiments:
        if exp.get("status") == "loading":
            exp_name = exp.get("name")
            metadata = processing_metadata.get(exp_name, {})
            start_time_str = metadata.get("start_time")
            
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    elapsed_seconds = (current_time - start_time).total_seconds()
                    
                    if elapsed_seconds > EXPERIMENT_TIMEOUT_SECONDS:
                        stuck_experiments.append(exp_name)
                        logger.warning(f"Experiment '{exp_name}' stuck in loading for {elapsed_seconds:.0f}s (timeout: {EXPERIMENT_TIMEOUT_SECONDS}s)")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing start time for {exp_name}: {e}")
    
    if not stuck_experiments:
        raise PreventUpdate
    
    # Remove stuck experiments
    logger.info(f"Cleaning up {len(stuck_experiments)} stuck experiments: {stuck_experiments}")
    updated_experiments = [
        exp for exp in experiments
        if exp.get("name") not in stuck_experiments
    ]
    
    # Clear metadata for stuck experiments
    for exp_name in stuck_experiments:
        if exp_name in processing_metadata:
            processing_metadata[exp_name] = {
                "start_time": processing_metadata[exp_name].get("start_time"),
                "status": "timeout",
                "end_time": current_time.isoformat(),
            }
    
    # Save to storage (DO NOT use session - too large for cookies)
    if has_request_context():
        user_key = session.get("user", "default")
        save_experiments(user_key, updated_experiments)
        logger.info(f"Saved cleanup results for user '{user_key}'")
    
    return updated_experiments, processing_metadata


@app.callback(
    Output("attachment-area", "style"),
    Output("attachment-area", "className"),
    Output("error-notification", "children"),
    Output("error-notification", "style"),
    Input("toggle-attachments", "n_clicks"),
    Input("composer-send", "n_clicks"),
    Input("selected-experiment", "data"),
    State("attachment-area", "style"),
    State("attachment-area", "className"),
)
def toggle_attachment_area(
    toggle_clicks: Optional[int],
    send_clicks: Optional[int],
    selection: Optional[str],
    current_style: Dict[str, Any],
    current_class: str,
):
    triggered = callback_context.triggered_id
    style = toggle_attachment_area_logic(
        triggered,
        toggle_clicks,
        selection or "New Experiment",
        viewport_height=None,
        composer_y=None,
    )

    # Clear error when toggling attachment or switching experiments
    clear_error = triggered in ["toggle-attachments", "selected-experiment"]

    if style.get("display") == "none":
        return (
            {"display": "none"},
            "attachment-popover attachment-bottom",
            None if clear_error else html.Div(),
            {"display": "none"} if clear_error else {},
        )

    class_name = style.get(
        "className",
        current_class or "attachment-popover attachment-bottom",
    )
    return (
        {"display": "block"},
        class_name,
        None if clear_error else html.Div(),
        {"display": "none"} if clear_error else {},
    )


@app.callback(
    Output("error-message", "data", allow_duplicate=True),
    Input("delete-confirmation-confirm", "n_clicks"),
    prevent_initial_call=True,
)
def debug_delete_button_click(n_clicks):
    """Debug callback to verify Delete button clicks are being registered."""
    logger.info(f"🔍 DEBUG: Delete button clicked! n_clicks={n_clicks}")
    logger.info(f"🔍 DEBUG: Callback triggered successfully!")
    return None  # Don't actually use this output


@app.callback(
    Output("delete-confirmation-experiment", "data", allow_duplicate=True),
    Output("delete-confirmation-visible", "data", allow_duplicate=True),
    Input({"type": "experiment-menu-button", "experiment": ALL}, "n_clicks"),
    State({"type": "experiment-menu-button", "experiment": ALL}, "id"),
    prevent_initial_call=True,
)
def show_delete_confirmation(clicks, button_ids):
    """Show delete confirmation when menu button is clicked."""
    triggered = callback_context.triggered_id
    
    logger.debug(f"show_delete_confirmation triggered: {triggered}")
    logger.debug(f"Clicks received: {clicks}")
    logger.debug(f"Button IDs: {button_ids}")
    
    # Validate trigger exists and is a dict
    if not triggered or not isinstance(triggered, dict):
        logger.debug("No valid trigger or not a dict, preventing update")
        raise PreventUpdate
    
    # Validate it's the right button type
    if triggered.get("type") != "experiment-menu-button":
        logger.debug(f"Wrong button type: {triggered.get('type')}, preventing update")
        raise PreventUpdate
    
    # Check if any actual clicks occurred (avoid initial state where all are 0)
    if not any(c for c in clicks if c and c > 0):
        logger.debug("No actual clicks detected (all 0 or None)")
        raise PreventUpdate
    
    experiment_name = triggered.get("experiment")
    if not experiment_name:
        logger.debug("No experiment name in trigger")
        raise PreventUpdate
    
    logger.info(f"Showing delete confirmation for experiment: {experiment_name}")
    return experiment_name, True


@app.callback(
    Output("delete-confirmation-modal", "children"),
    Output("delete-confirmation-modal", "style"),
    Input("delete-confirmation-visible", "data"),
    Input("delete-confirmation-experiment", "data"),
)
def render_delete_confirmation(visible, experiment_name):
    """Render the delete confirmation modal."""
    logger.debug(f"render_delete_confirmation called: visible={visible}, experiment={experiment_name}")
    
    if not visible or not experiment_name:
        logger.debug("Modal hidden (not visible or no experiment name)")
        return None, {"display": "none"}
    
    logger.info(f"Rendering delete confirmation modal for: {experiment_name}")
    
    modal = html.Div(
        html.Div(
            [
                html.Div("Delete Experiment?", className="delete-confirmation-title"),
                html.Div(
                    f'Are you sure you want to delete "{experiment_name}"? This action cannot be undone.',
                    className="delete-confirmation-message",
                ),
                html.Div(
                    [
                        html.Button(
                            "Cancel",
                            id="delete-confirmation-cancel",
                            className="delete-confirmation-button cancel",
                            n_clicks=0,
                            **{"type": "button"},
                        ),
                        html.Button(
                            "Delete",
                            id="delete-confirmation-confirm",
                            className="delete-confirmation-button confirm",
                            n_clicks=0,
                            **{"type": "button"},
                        ),
                    ],
                    className="delete-confirmation-buttons",
                ),
            ],
            className="delete-confirmation-modal",
        ),
        className="delete-confirmation-overlay",
    )
    
    return modal, {"display": "block"}


@app.callback(
    Output("experiments-store", "data", allow_duplicate=True),
    Output("selected-experiment", "data", allow_duplicate=True),
    Output("planning-sessions-store", "data", allow_duplicate=True),
    Output("selected-planning-session", "data", allow_duplicate=True),
    Output("delete-confirmation-experiment", "data", allow_duplicate=True),
    Output("delete-confirmation-visible", "data", allow_duplicate=True),
    Input("delete-confirmation-confirm", "n_clicks"),
    Input("delete-confirmation-cancel", "n_clicks"),
    State("delete-confirmation-experiment", "data"),
    State("experiments-store", "data"),
    State("selected-experiment", "data"),
    State("planning-sessions-store", "data"),
    State("selected-planning-session", "data"),
    State("app-mode", "data"),
    prevent_initial_call=True,
)
def handle_delete_confirmation_unified(confirm_clicks, cancel_clicks, item_name, experiments, selected_exp, planning_sessions, selected_planning, mode):
    """Handle delete confirmation or cancellation for both experiments and planning sessions."""
    from dash import no_update
    
    # Prevent callback from firing when modal first renders
    if not confirm_clicks and not cancel_clicks:
        logger.debug("Modal rendered but no button clicked yet, preventing update")
        raise PreventUpdate
    
    triggered = callback_context.triggered_id
    
    logger.info(f"=== handle_delete_confirmation_unified CALLED ===")
    logger.info(f"Triggered ID: {triggered}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Item to delete: {item_name}")
    
    if triggered == "delete-confirmation-cancel":
        logger.info("Delete cancelled by user")
        # Close the modal, don't update any data
        return no_update, no_update, no_update, no_update, None, False
    
    if triggered == "delete-confirmation-confirm":
        if not item_name:
            logger.error("Confirm clicked but no item name provided!")
            raise PreventUpdate
        
        if mode == "planning":
            # Handle planning session deletion
            logger.info(f"Deleting planning session: {item_name}")
            
            sessions = planning_sessions or []
            updated_sessions = [s for s in sessions if s.get("name") != item_name]
            
            logger.info(f"Sessions before delete: {len(sessions)}, after delete: {len(updated_sessions)}")
            
            # Save to storage
            if has_request_context():
                user_key = session.get("user", "default")
                save_planning_sessions(user_key, updated_sessions)
                logger.info(f"Deleted planning session '{item_name}' for user '{user_key}'")
            
            # If deleted session was selected, switch to "New Planning"
            new_planning_selection = "New Planning" if selected_planning == item_name else selected_planning
            
            # Return: experiments unchanged, planning updated, modal closed
            return no_update, no_update, updated_sessions, new_planning_selection, None, False
        
        else:  # mode == "experiments" or default
            # Handle experiment deletion
            logger.info(f"Deleting experiment: {item_name}")
            
            experiments = experiments or []
            updated_experiments = [exp for exp in experiments if exp.get("name") != item_name]
            
            logger.info(f"Experiments before delete: {len(experiments)}, after delete: {len(updated_experiments)}")
            
            # Save to storage
            if has_request_context():
                user_key = session.get("user", "default")
                save_experiments(user_key, updated_experiments)
                logger.info(f"Deleted experiment '{item_name}' for user '{user_key}'")
            
            # If deleted experiment was selected, switch to "New Experiment"
            new_exp_selection = "New Experiment" if selected_exp == item_name else selected_exp
            
            # Return: experiments updated, planning unchanged, modal closed
            return updated_experiments, new_exp_selection, no_update, no_update, None, False
    
    logger.debug(f"No valid trigger ({triggered}), preventing update")
    raise PreventUpdate


@app.callback(
    Output("chat-panel", "className"),
    Output("chat-visible", "data"),
    Input("chat-toggle-button", "n_clicks"),
    Input("chat-close-button", "n_clicks"),
    State("chat-visible", "data"),
)
def toggle_chat_panel(toggle_clicks, close_clicks, is_visible):
    """Toggle chat panel visibility."""
    triggered = callback_context.triggered_id
    
    if not triggered:
        return "chat-panel", False
    
    if triggered == "chat-toggle-button":
        # Open the panel
        return "chat-panel open", True
    elif triggered == "chat-close-button":
        # Close the panel
        return "chat-panel", False
    
    return "chat-panel", False


@app.callback(
    Output("chat-toggle-button", "style"),
    Input("selected-experiment", "data"),
    State("experiments-store", "data"),
)
def show_chat_button(selection, experiments):
    """Show chat button only for completed experiments."""
    if not selection or selection == "New Experiment":
        return {"display": "none"}
    
    experiments = experiments or []
    
    # Find the selected experiment
    for exp in experiments:
        if exp.get("name") == selection:
            # Show button only if experiment is complete
            if exp.get("status") == "complete":
                return {"display": "flex"}
            break
    
    return {"display": "none"}


@app.callback(
    Output("chat-messages", "data", allow_duplicate=True),
    Output("chat-validation-error", "data", allow_duplicate=True),
    Input("selected-experiment", "data"),
    prevent_initial_call=True,
)
def reset_chat_on_experiment_change(selection):
    """Reset chat when switching experiments."""
    logger.info(f"Resetting chat for experiment: {selection}")
    return [], None


@app.callback(
    Output("chat-messages", "data", allow_duplicate=True),
    Output("chat-input", "value"),
    Output("chat-loading", "data"),
    Output("chat-validation-error", "data", allow_duplicate=True),
    Input("chat-send-button", "n_clicks"),
    State("chat-input", "value"),
    State("chat-messages", "data"),
    State("selected-experiment", "data"),
    State("experiments-store", "data"),
    prevent_initial_call=True,
)
def handle_chat_message(n_clicks, user_input, messages, selected_exp, experiments):
    """Handle chat message submission with validation and LLM response."""
    if not n_clicks or not user_input or not user_input.strip():
        raise PreventUpdate
    
    logger.info(f"Processing chat message for experiment: {selected_exp}")
    
    messages = messages or []
    experiments = experiments or []
    
    # Find the current experiment
    current_experiment = None
    for exp in experiments:
        if exp.get("name") == selected_exp:
            current_experiment = exp
            break
    
    if not current_experiment:
        logger.error(f"Experiment not found: {selected_exp}")
        return messages, user_input, False, "Experiment not found."
    
    # Get OpenAI client
    client = current_app.config.get("OPENAI_CLIENT")
    if not client:
        logger.error("OpenAI client not configured")
        return messages, user_input, False, "Chat service is not available."
    
    # Validate the question
    from services.chat_validator import ChatValidator
    validator = ChatValidator(client)
    
    try:
        should_continue, reason = validator.validate_question(user_input)
        
        if not should_continue:
            logger.info(f"Question rejected: {reason}")
            # Keep the input, show error
            return messages, user_input, False, reason
        
        logger.info("Question validated successfully")
        
        # Generate response with experiment context
        from services.experiment_chat import ExperimentChatService
        chat_service = ExperimentChatService(client)
        
        assistant_response = chat_service.chat_with_context(
            current_experiment,
            messages,
            user_input
        )
        
        # Add both messages to history
        updated_messages = messages + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ]
        
        logger.info(f"Chat response generated, total messages: {len(updated_messages)}")
        
        # Clear input, clear error
        return updated_messages, "", False, None
        
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        logger.error(traceback.format_exc())
        return messages, user_input, False, f"An error occurred: {str(e)}"


@app.callback(
    Output("chat-messages-container", "children"),
    Input("chat-messages", "data"),
)
def render_chat_messages(messages):
    """Render chat messages."""
    if not messages:
        # Show empty state
        return [
            html.Div(
                [
                    html.Div("💬", className="chat-empty-state-icon"),
                    html.Div(
                        "Ask me anything about this experiment!",
                        style={"fontSize": "1.125rem", "fontWeight": "600", "marginBottom": "0.5rem"}
                    ),
                    html.Div(
                        "I can help explain the results, methodology, and statistical concepts.",
                        style={"fontSize": "0.875rem"}
                    ),
                ],
                className="chat-empty-state",
            )
        ]
    
    message_components = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        is_user = role == "user"
        
        # Create avatar
        avatar = html.Div(
            "👤" if is_user else "🤖",
            className=f"chat-avatar {'user' if is_user else 'assistant'}",
        )
        
        # Create message bubble - use Markdown for assistant, plain text for user
        if is_user:
            bubble = html.Div(
                content,
                className="chat-bubble user",
            )
        else:
            bubble = dcc.Markdown(
                content,
                className="chat-bubble assistant",
            )
        
        # Create message container
        message_div = html.Div(
            [avatar, bubble],
            className=f"chat-message {'user' if is_user else 'assistant'}",
        )
        
        message_components.append(message_div)
    
    return message_components


@app.callback(
    Output("chat-input", "disabled"),
    Output("chat-send-button", "disabled"),
    Input("chat-loading", "data"),
)
def update_loading_state(is_loading):
    """Update loading state for input and button."""
    return is_loading, is_loading


@app.callback(
    Output("chat-validation-error-display", "children"),
    Output("chat-validation-error-display", "style"),
    Input("chat-validation-error", "data"),
)
def display_validation_error(error):
    """Display validation error message."""
    if not error:
        return None, {"display": "none"}
    
    return html.Div(
        [
            html.Span("⚠", style={"fontSize": "1rem"}),
            html.Span(error),
        ],
        className="validation-error",
    ), {"display": "flex"}


# ============================================================================
# PLANNING MODE CALLBACKS
# ============================================================================

@app.callback(
    Output("app-mode", "data"),
    Output("mode-experiments", "className"),
    Output("mode-planning", "className"),
    Input("mode-experiments", "n_clicks"),
    Input("mode-planning", "n_clicks"),
    State("app-mode", "data"),
)
def switch_mode(exp_clicks, plan_clicks, current_mode):
    """Switch between Experiments and Planning modes."""
    triggered = callback_context.triggered_id
    
    if not triggered:
        # Initial load - set classes based on current mode
        if current_mode == "planning":
            return "planning", "mode-button", "mode-button active"
        return "experiments", "mode-button active", "mode-button"
    
    if triggered == "mode-experiments":
        logger.info("Switching to Experiments mode")
        return "experiments", "mode-button active", "mode-button"
    elif triggered == "mode-planning":
        logger.info("Switching to Planning mode")
        return "planning", "mode-button", "mode-button active"
    
    # Default to experiments
    return "experiments", "mode-button active", "mode-button"


@app.callback(
    Output("selected-planning-session", "data"),
    Input({"type": "planning-card", "value": ALL}, "n_clicks"),
    State({"type": "planning-card", "value": ALL}, "id"),
)
def select_planning_session(cards_clicks, card_ids):
    """Select a planning session from the sidebar."""
    triggered = callback_context.triggered_id
    
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate
    
    session_name = triggered.get("value", "New Planning")
    logger.info(f"Selected planning session: {session_name}")
    return session_name


@app.callback(
    Output("planning-messages", "data", allow_duplicate=True),
    Input("selected-planning-session", "data"),
    State("planning-sessions-store", "data"),
    prevent_initial_call=True,
)
def load_planning_messages(session_name, sessions):
    """Load messages when a planning session is selected."""
    if not session_name or session_name == "New Planning":
        return []
    
    sessions = sessions or []
    for session in sessions:
        if session.get("name") == session_name:
            return session.get("messages", [])
    
    return []


@app.callback(
    Output("planning-messages-container", "children"),
    Input("planning-messages", "data"),
)
def render_planning_messages(messages):
    """Render planning messages."""
    if not messages:
        return html.Div(
            [
                html.Div("💭", style={"fontSize": "3rem", "marginBottom": "1rem"}),
                # html.Div(
                #     "Start planning your experiment!",
                #     style={"fontSize": "1.125rem", "fontWeight": "600", "marginBottom": "0.5rem", "color": "#64748b"}
                # ),
                html.Div(
                    "Ask about experimental design, methodology, or causal inference.",
                    style={"fontSize": "0.875rem", "color": "#94a3b8"}
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "minHeight": "200px"}
        )
    
    message_components = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        is_user = role == "user"
        
        # Create avatar
        avatar = html.Div(
            "👤" if is_user else "🤖",
            className=f"chat-avatar {'user' if is_user else 'assistant'}",
        )
        
        # Create message bubble - use Markdown for assistant, plain text for user
        if is_user:
            bubble = html.Div(
                content,
                className="planning-bubble user",
            )
        else:
            bubble = dcc.Markdown(
                content,
                className="planning-bubble assistant",
            )
        
        # Create message container with avatar
        message_div = html.Div(
            [avatar, bubble],
            className=f"planning-message {'user' if is_user else 'assistant'}",
        )
        
        message_components.append(message_div)
    
    return message_components


@app.callback(
    Output("planning-sessions-store", "data", allow_duplicate=True),
    Output("selected-planning-session", "data", allow_duplicate=True),
    Output("planning-messages", "data", allow_duplicate=True),
    Output("planning-input", "value"),
    Output("planning-validation-error", "data"),
    Input("planning-send", "n_clicks"),
    State("planning-input", "value"),
    State("planning-messages", "data"),
    State("selected-planning-session", "data"),
    State("planning-sessions-store", "data"),
    prevent_initial_call=True,
)
def handle_planning_message(n_clicks, user_input, messages, selected_session, sessions):
    """Handle planning message submission with validation and LLM response."""
    if not n_clicks or not user_input or not user_input.strip():
        raise PreventUpdate
    
    logger.info(f"Processing planning message for session: {selected_session}")
    
    messages = messages or []
    sessions = sessions or []
    
    # Get OpenAI client
    client = current_app.config.get("OPENAI_CLIENT")
    if not client:
        logger.error("OpenAI client not configured")
        return sessions, selected_session, messages, user_input, "Chat service is not available."
    
    # Validate the question
    from services.planning_validator import PlanningValidator
    validator = PlanningValidator(client)
    
    try:
        should_continue, reason = validator.validate_question(user_input)
        
        if not should_continue:
            logger.info(f"Question rejected: {reason}")
            # Keep the input, show error
            return sessions, selected_session, messages, user_input, reason
        
        logger.info("Question validated successfully")
        
        # Generate response
        from services.planning_chat import PlanningChatService
        chat_service = PlanningChatService(client)
        
        assistant_response = chat_service.chat(messages, user_input)
        
        # Add both messages to history
        updated_messages = messages + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ]
        
        # Create or update session
        if selected_session == "New Planning":
            # Create new session
            session_name = _generate_planning_session_name(len(sessions) + 1)
            new_session = {
                "name": session_name,
                "messages": updated_messages,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            updated_sessions = sessions + [new_session]
            selected_session = session_name
        else:
            # Update existing session
            updated_sessions = []
            for sess in sessions:
                if sess.get("name") == selected_session:
                    sess["messages"] = updated_messages
                    sess["updated_at"] = datetime.now().isoformat()
                updated_sessions.append(sess)
        
        # Save to storage
        if has_request_context():
            user_key = session.get("user", "default")
            save_planning_sessions(user_key, updated_sessions)
            logger.info(f"Saved planning sessions for user '{user_key}'")
        
        logger.info(f"Planning chat response generated, total messages: {len(updated_messages)}")
        
        # Clear input, clear error
        return updated_sessions, selected_session, updated_messages, "", None
        
    except Exception as e:
        logger.error(f"Error handling planning message: {e}")
        logger.error(traceback.format_exc())
        return sessions, selected_session, messages, user_input, f"An error occurred: {str(e)}"


@app.callback(
    Output("planning-validation-error-display", "children"),
    Output("planning-validation-error-display", "style"),
    Input("planning-validation-error", "data"),
)
def display_planning_validation_error(error):
    """Display planning validation error message."""
    if not error:
        return None, {"display": "none"}
    
    return html.Div(
        [
            html.Span("⚠", style={"fontSize": "1rem"}),
            html.Span(error),
        ],
        className="validation-error",
        style={"color": "#ef4444", "fontSize": "0.75rem", "marginTop": "0.5rem", "display": "flex", "alignItems": "center", "gap": "0.25rem"}
    ), {"display": "flex"}


@app.callback(
    Output("delete-confirmation-experiment", "data", allow_duplicate=True),
    Output("delete-confirmation-visible", "data", allow_duplicate=True),
    Input({"type": "planning-menu-button", "session": ALL}, "n_clicks"),
    State({"type": "planning-menu-button", "session": ALL}, "id"),
    prevent_initial_call=True,
)
def show_delete_planning_confirmation(clicks, button_ids):
    """Show delete confirmation when planning session menu button is clicked."""
    triggered = callback_context.triggered_id
    
    logger.debug(f"show_delete_planning_confirmation triggered: {triggered}")
    
    # Validate trigger exists and is a dict
    if not triggered or not isinstance(triggered, dict):
        logger.debug("No valid trigger or not a dict, preventing update")
        raise PreventUpdate
    
    # Validate it's the right button type
    if triggered.get("type") != "planning-menu-button":
        logger.debug(f"Wrong button type: {triggered.get('type')}, preventing update")
        raise PreventUpdate
    
    # Check if any actual clicks occurred
    if not any(c for c in clicks if c and c > 0):
        logger.debug("No actual clicks detected (all 0 or None)")
        raise PreventUpdate
    
    session_name = triggered.get("session")
    if not session_name:
        logger.debug("No session name in trigger")
        raise PreventUpdate
    
    logger.info(f"Showing delete confirmation for planning session: {session_name}")
    return session_name, True


if __name__ == "__main__":
    app.run(debug=True)



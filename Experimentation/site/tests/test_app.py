"""Unit tests for the mock experimentation Dash app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
from dash import html

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import (  # type: ignore  # noqa: E402
    build_dashboard,
    build_placeholder_dashboard,
    generate_placeholder_figure,
    manage_experiments,
    populate_experiment_view,
)


def _sample_experiment() -> Dict[str, Optional[str]]:
    return {
        "id": "exp-123",
        "name": "Experiment 1",
        "message": "Run lift analysis in DE",
        "start_date": "2025-04-01",
        "end_date": "2025-04-15",
        "covariates": "price,discount",
        "dataset_name": "metrics.csv",
        "created_at": "Apr 10, 2025 18:30 UTC",
    }


def test_generate_placeholder_figure_structure() -> None:
    figure = generate_placeholder_figure()

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert {trace.name for trace in figure.data} == {"Observed", "Counterfactual"}


def test_build_dashboard_contains_expected_sections() -> None:
    experiment = _sample_experiment()

    dashboard = build_dashboard(experiment)

    assert isinstance(dashboard, html.Div)
    assert any(
        isinstance(child, html.Div) and "Experiment Insights" in child.children[0].children
        for child in dashboard.children
    )
    attachment_section = dashboard.children[1]
    assert len(attachment_section.children) == 5


def test_build_placeholder_dashboard_prompts_user() -> None:
    placeholder = build_placeholder_dashboard()

    assert isinstance(placeholder, html.Div)
    texts = [child.children for child in placeholder.children if isinstance(child, html.P)]
    assert any("Describe the experimental request" in text for text in texts)


def test_manage_experiments_creates_and_returns_new_experiment() -> None:
    ctx_stub = MagicMock()
    ctx_stub.triggered = [{"prop_id": "run-experiment-btn.n_clicks"}]
    ctx_stub.triggered_id = "run-experiment-btn"

    with patch("app.dash.callback_context", ctx_stub):
        store, options, active_id = manage_experiments(
            run_clicks=1,
            selected_experiment=None,
            sidebar_new_clicks=None,
            card_clicks=[],
            store={"experiments": [], "active_id": None},
            message="Check uplift",
            start_date="2025-04-01",
            end_date="2025-04-15",
            covariates="channel_spend",
            dataset_filename="uplift.csv",
        )

    assert active_id is not None
    assert options and options[0]["label"] == "Experiment 1"
    assert store["experiments"][0]["dataset_name"] == "uplift.csv"


def test_populate_experiment_view_placeholder_response() -> None:
    ctx_stub = MagicMock()
    ctx_stub.triggered = []
    ctx_stub.triggered_id = None

    with patch("app.dash.callback_context", ctx_stub):
        (
            prompt_value,
            start_date,
            end_date,
            covariates,
            dashboard,
            heading,
            dataset_label,
            summaries,
            sidebar_class,
            content_class,
            ui_state,
        ) = populate_experiment_view(
            store={"experiments": [], "active_id": None},
            uploaded_filename=None,
            ui_state={"sidebar_collapsed": False},
        )

    assert prompt_value == ""
    assert start_date is None and end_date is None
    assert covariates == ""
    assert heading == "New Experiment"
    assert dataset_label == "No dataset attached"
    assert isinstance(dashboard, html.Div)
    assert isinstance(summaries, list)
    assert sidebar_class == "sidebar-column expanded"
    assert content_class == "content-column"
    assert ui_state == {"sidebar_collapsed": False}


def test_populate_experiment_view_existing_experiment() -> None:
    experiment = _sample_experiment()
    store = {"experiments": [experiment], "active_id": experiment["id"]}

    ctx_stub = MagicMock()
    ctx_stub.triggered = []
    ctx_stub.triggered_id = None

    with patch("app.dash.callback_context", ctx_stub):
        (
            prompt_value,
            start_date,
            end_date,
            covariates,
            dashboard,
            heading,
            dataset_label,
            summaries,
            sidebar_class,
            content_class,
            ui_state,
        ) = populate_experiment_view(
            store=store,
            uploaded_filename=None,
            ui_state={"sidebar_collapsed": False},
        )

    assert prompt_value == experiment["message"]
    assert start_date == experiment["start_date"]
    assert end_date == experiment["end_date"]
    assert covariates == experiment["covariates"]
    assert heading == f"Viewing {experiment['name']}"
    assert dataset_label == f"Attached: {experiment['dataset_name']}"
    assert isinstance(dashboard, html.Div)
    assert len(summaries) == 1
    assert sidebar_class == "sidebar-column expanded"
    assert content_class == "content-column"
    assert ui_state == {"sidebar_collapsed": False}


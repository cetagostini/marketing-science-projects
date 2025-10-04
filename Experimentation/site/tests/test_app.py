"""Smoke tests for the mock experimentation Dash app."""

from dash import dcc, html

from site.app import (
    _make_experiment_name,
    app,
    create_dashboard_components,
)


def test_make_experiment_name_truncates_and_formats():
    """The preview suffix should be trimmed to 24 characters with collapsed whitespace."""

    message = "   Multi   spaced   message to ensure trimming works properly  "
    result = _make_experiment_name(5, message)

    assert result == "Experiment 5: Multi spaced message to"


def test_make_experiment_name_defaults_without_message():
    """If no previewable text is provided the counter alone should be used."""

    assert _make_experiment_name(2, "") == "Experiment 2"
    assert _make_experiment_name(3, None) == "Experiment 3"


def test_create_dashboard_components_structure():
    """The dashboard wrapper should include key sections and echo back experiment details."""

    experiment = {
        "message": "Test prompt",
        "start_date": "2025-04-01",
        "end_date": "2025-04-15",
        "dataset": "EU Retail",
        "covariates": "Seasonality",
    }

    dashboard = create_dashboard_components(experiment)

    assert isinstance(dashboard, html.Div)
    assert dashboard.className == "dashboard-wrapper"

    child_classes = [getattr(child, "className", "") for child in dashboard.children]
    assert "figure-row" in child_classes
    assert "panel-grid" in child_classes

    panel_grid = next(child for child in dashboard.children if getattr(child, "className", "") == "panel-grid")
    experiment_summary = next(
        child for child in panel_grid.children if getattr(child, "className", "") == "experiment-brief"
    )

    prompt_section = experiment_summary.children[1]
    prompt_value = prompt_section.children[1].children
    assert prompt_value == "Test prompt"


def test_app_layout_and_callbacks_registered():
    """Validate the top-level layout wiring and callback registry."""

    layout = app.layout
    assert isinstance(layout, html.Div)

    stores = [child for child in layout.children if isinstance(child, dcc.Store)]
    store_ids = {store.id for store in stores}
    assert store_ids == {"experiments-store", "attachments-store"}

    assert len(app.callback_map) == 5

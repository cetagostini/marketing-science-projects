"""Unit tests for the experimentation Dash app internals."""

import os
import sys
from types import SimpleNamespace

import pytest
from dash import dcc, html

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Experimentation.site import app as app_module
from Experimentation.site.app import (
    _make_experiment_name,
    app,
    create_dashboard_components,
    handle_experiment_actions,
    refresh_experiment_list,
    render_dashboard,
    set_attachment_panel,
    toggle_attachments,
)


@pytest.fixture
def experiment_payload():
    return {
        "message": "Measure conversion lift",
        "start_date": "2025-04-01",
        "end_date": "2025-04-15",
        "dataset": "EU Retail",
        "covariates": "Seasonality, Spend",
    }


@pytest.fixture
def fresh_store():
    return {"experiments": [], "counter": 0, "selected_id": None}


@pytest.mark.parametrize(
    ("counter", "message", "expected"),
    [
        (5, "   Multi   spaced   message to ensure trimming works properly  ", "Experiment 5: Multi spaced message to"),
        (2, "", "Experiment 2"),
        (3, None, "Experiment 3"),
    ],
)
def test_make_experiment_name_formats(counter, message, expected):
    assert _make_experiment_name(counter, message) == expected


def test_create_dashboard_components_sections(experiment_payload):
    dashboard = create_dashboard_components(experiment_payload)

    assert isinstance(dashboard, html.Div)
    assert dashboard.className == "dashboard-wrapper"

    class_names = {getattr(child, "className", "") for child in dashboard.children}
    assert {"dashboard-cards", "figure-row", "panel-grid"} <= class_names

    figure_row = next(child for child in dashboard.children if getattr(child, "className", "") == "figure-row")
    graphs = [
        grandchild
        for card in figure_row.children
        for grandchild in getattr(card, "children", [])
        if isinstance(grandchild, dcc.Graph)
    ]
    assert len(graphs) == 2

    panel_grid = next(child for child in dashboard.children if getattr(child, "className", "") == "panel-grid")
    experiment_brief = next(child for child in panel_grid.children if getattr(child, "className", "") == "experiment-brief")
    prompt_section = experiment_brief.children[1]
    assert prompt_section.children[1].children == experiment_payload["message"]


def test_toggle_attachments_toggles_open_flag():
    assert toggle_attachments(1, {"open": False}) == {"open": True}
    assert toggle_attachments(2, {"open": True}) == {"open": False}
    assert toggle_attachments(3, {}) == {"open": True}


def test_set_attachment_panel_respects_open_state():
    assert set_attachment_panel({"open": True}) == "attachment-panel"
    assert set_attachment_panel({"open": False}) == "attachment-panel hidden"
    assert set_attachment_panel({}) == "attachment-panel hidden"


def test_refresh_experiment_list_with_experiments():
    store_data = {
        "experiments": [
            {
                "id": "exp-01",
                "name": "Experiment 1",
                "start_date": "2025-04-01",
                "end_date": "2025-04-15",
            }
        ]
    }

    options, children = refresh_experiment_list(store_data)

    assert options == [{"label": "Experiment 1", "value": "exp-01"}]
    assert isinstance(children, list)
    assert children[0].className == "sidebar-experiment"


def test_refresh_experiment_list_without_experiments():
    options, children = refresh_experiment_list({"experiments": []})

    assert options == []
    assert isinstance(children, html.Div)
    assert children.className == "sidebar-empty"


def test_handle_experiment_actions_creates_new_experiment(monkeypatch, fresh_store, experiment_payload):
    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="run-experiment-button"))

    result = handle_experiment_actions(
        run_clicks=1,
        new_clicks=None,
        message=experiment_payload["message"],
        start_date=experiment_payload["start_date"],
        end_date=experiment_payload["end_date"],
        dataset=experiment_payload["dataset"],
        covariates=experiment_payload["covariates"],
        store_data=fresh_store,
    )

    store_after, selected_id, message_value, start_date, end_date, dataset, covariates = result

    assert store_after["counter"] == 1
    assert len(store_after["experiments"]) == 1

    experiment = store_after["experiments"][0]
    assert experiment["message"] == experiment_payload["message"]
    assert experiment["start_date"] == experiment_payload["start_date"]
    assert experiment["dataset"] == experiment_payload["dataset"]
    assert experiment["covariates"] == experiment_payload["covariates"]
    assert isinstance(experiment["id"], str)
    assert isinstance(experiment["created_at"], str)

    assert selected_id == experiment["id"]
    assert message_value == ""
    assert start_date is None
    assert end_date is None
    assert dataset == ""
    assert covariates == ""


def test_handle_experiment_actions_ignores_empty_submission(monkeypatch, fresh_store):
    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="run-experiment-button"))

    result = handle_experiment_actions(
        run_clicks=1,
        new_clicks=None,
        message="",
        start_date=None,
        end_date=None,
        dataset="",
        covariates="",
        store_data=fresh_store,
    )

    assert result[0] is fresh_store
    assert result[1] == fresh_store["selected_id"]


def test_handle_experiment_actions_new_button_resets_form(monkeypatch):
    store_data = {"experiments": [], "counter": 2, "selected_id": "exp-02"}
    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="new-experiment-button"))

    result = handle_experiment_actions(
        run_clicks=None,
        new_clicks=1,
        message="Keep existing",
        start_date="2025-04-01",
        end_date="2025-04-15",
        dataset="EU Retail",
        covariates="Seasonality",
        store_data=store_data,
    )

    returned_store, selector_value, message_value, start_date, end_date, dataset, covariates = result

    assert returned_store["selected_id"] is None
    assert selector_value is None
    assert message_value == ""
    assert start_date is None
    assert end_date is None
    assert dataset == ""
    assert covariates == ""


def test_render_dashboard_empty_state(fresh_store):
    component = render_dashboard(selected_id=None, store_data=fresh_store)

    assert isinstance(component, html.Div)
    assert component.className == "dashboard-empty"


def test_render_dashboard_selected_experiment(monkeypatch):
    experiment = {"id": "exp-01"}
    store_data = {"experiments": [experiment]}
    sentinel = html.Div(id="sentinel-dashboard")

    monkeypatch.setattr(app_module, "create_dashboard_components", lambda exp: sentinel)

    component = render_dashboard(selected_id="exp-01", store_data=store_data)

    assert component is sentinel


def test_app_layout_contains_expected_sections():
    layout = app.layout

    assert isinstance(layout, html.Div)
    store_ids = {child.id for child in layout.children if isinstance(child, dcc.Store)}
    assert store_ids == {"experiments-store", "attachments-store"}
    assert app.title == "Experimentation Studio"

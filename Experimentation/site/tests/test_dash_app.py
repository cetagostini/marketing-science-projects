"""Tests for the Dash experimentation console app."""

from __future__ import annotations

import sys
import shutil
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pytest

from dash.testing.composite import DashComposite

from dash import Dash, no_update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("APP_SECRET", "test-secret")
os.environ.setdefault("APP_USERS", "alice:password")

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:  # pragma: no cover
    pass

import dash_app  # noqa: E402  pylint: disable=wrong-import-position
from app import server  # noqa: E402  pylint: disable=wrong-import-position
from storage import load_experiments, save_experiments


HAS_CHROMEDRIVER = shutil.which("chromedriver") is not None


@pytest.fixture
def dash_app_instance() -> Dash:
    """Provide a Dash app instance for use with dash_duo."""

    return dash_app.app


@pytest.fixture
def flask_client():
    server.config.update(SECRET_KEY="test-secret", TESTING=True)
    with server.test_client() as client:
        with client.session_transaction() as sess:
            sess.clear()
        yield client


@pytest.fixture
def login_dash_user(dash_app_instance: Dash, dash_duo: DashComposite) -> None:
    dash_duo.start_server(dash_app_instance)

    dash_duo.driver.get(dash_duo.server_url + "login")
    dash_duo.wait_for_element("#username").send_keys("alice")
    dash_duo.wait_for_element("#password").send_keys("password")
    dash_duo.wait_for_element("button[type='submit']").click()

    dash_duo.wait_for_page("/dash/")


skip_no_chromedriver = pytest.mark.skipif(
    not HAS_CHROMEDRIVER,
    reason="chromedriver executable not found in PATH; install ChromeDriver to run Dash UI tests",
)


@skip_no_chromedriver
def test_initial_layout(dash_duo: DashComposite, dash_app_instance: Dash, login_dash_user) -> None:
    dash_duo.wait_for_text_to_equal("#top-title", "New Experiment")
    dash_duo.wait_for_element("#new-experiment-view", timeout=2)

    cards = dash_duo.find_elements("css selector", ".experiment-card")
    assert len(cards) == 1
    assert "New Experiment" in cards[0].text

    message_field = dash_duo.find_element("#form-message")
    assert message_field.get_attribute("value") == ""

    assert dash_duo.find_element("#start-date input").get_attribute("value")
    assert dash_duo.find_element("#end-date input").get_attribute("value")


@skip_no_chromedriver
def test_run_experiment_flow(dash_duo: DashComposite, dash_app_instance: Dash, login_dash_user) -> None:
    message_field = dash_duo.find_element("#form-message")
    message_field.send_keys("Launch promo experiment")

    send_button = dash_duo.find_element("#composer-send")
    send_button.click()

    dash_duo.wait_for_element("#top-title")
    assert dash_duo.find_element("#top-title").text.startswith("Experiment 1")

    cards = dash_duo.find_elements("css selector", ".experiment-card")
    selected_cards = [card for card in cards if "active" in card.get_attribute("class")]
    assert any("Experiment 1" in card.text for card in selected_cards)

    assert dash_duo.find_element("#experiment-detail-view").is_displayed()
    metrics = dash_duo.find_elements("css selector", ".metric-grid .metric-card")
    assert len(metrics) == 3


@skip_no_chromedriver
def test_app_runs_with_dash_run_method(dash_duo: DashComposite, dash_app_instance: Dash, login_dash_user) -> None:
    # Ensure the server responded and rendered the layout
    dash_duo.wait_for_text_to_equal("#top-title", "New Experiment")

    # No further assertions needed; the primary goal is to ensure the updated
    # entry point (app.run) works without raising the obsolete attribute error.


@skip_no_chromedriver
def test_logout_link_redirects_to_login(dash_duo: DashComposite, dash_app_instance: Dash, login_dash_user) -> None:
    dash_duo.wait_for_element("#sidebar")
    logout_link = dash_duo.wait_for_element("a.logout-button")
    logout_link.click()

    dash_duo.wait_for_page("/login")
    assert dash_duo.find_element("#username")


def test_handle_run_or_reset_logic_adds_experiment() -> None:
    result = dash_app.handle_run_or_reset_logic(
        triggered="composer-send",
        submit_clicks=1,
        reset_clicks=None,
        experiments=[],
        message="New launch",
        start_date_value="2025-04-01",
        end_date_value="2025-04-15",
        covariates_value="Seasonality",
        upload_filename="metrics.csv",
    )

    experiments, selected, message, start, end, covariates, upload_contents, upload_filename = result
    assert len(experiments) == 1
    assert experiments[0]["name"] == selected
    assert message == ""
    assert covariates == []
    assert upload_contents is None
    assert upload_filename is None
    assert start <= end


def test_save_and_load_experiments(tmp_path, monkeypatch) -> None:
    storage_path = tmp_path / "experiments.json"
    monkeypatch.setenv("EXPERIMENT_STORE_PATH", str(storage_path))

    sample_payload = [
        {
            "name": "Experiment 1",
            "message": "Hello",
            "start_date": "2025-04-01",
        }
    ]
    save_experiments("alice", sample_payload)

    loaded = load_experiments("alice")
    assert loaded == sample_payload

    assert load_experiments("bob") == []


def test_toggle_attachment_area_logic() -> None:
    style = dash_app.toggle_attachment_area_logic(
        triggered="toggle-attachments",
        toggle_clicks=1,
        selection="New Experiment",
        viewport_height=None,
        composer_y=None,
    )
    assert style["display"] == "block"
    assert style["className"] == "attachment-popover attachment-top"

    style = dash_app.toggle_attachment_area_logic(
        triggered="composer-send",
        toggle_clicks=5,
        selection="New Experiment",
    )
    assert style == {"display": "none"}

    style = dash_app.toggle_attachment_area_logic(
        triggered="toggle-attachments",
        toggle_clicks=2,
        selection="New Experiment",
        viewport_height=None,
        composer_y=None,
    )
    assert style == {"display": "none"}

    style = dash_app.toggle_attachment_area_logic(
        triggered="toggle-attachments",
        toggle_clicks=1,
        selection="Experiment 1",
        viewport_height=800,
        composer_y=500,
    )
    assert style == {"display": "none"}


def test_compute_popover_position() -> None:
    assert dash_app.compute_popover_position(600, 400) == "attachment-top"
    assert dash_app.compute_popover_position(600, 200) == "attachment-bottom"
    assert dash_app.compute_popover_position(None, None) == "attachment-top"



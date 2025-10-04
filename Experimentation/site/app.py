"""Streamlit application for mock experimentation dashboard."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
import os

from flask import Flask, redirect, request, session, url_for
from login import login_bp
from storage import load_experiments


server = Flask(__name__)
server.secret_key = os.getenv("APP_SECRET", "dev-secret")
server.register_blueprint(login_bp)
server.config["BOOT_TOKEN"] = os.getenv("APP_BOOT_TOKEN") or os.urandom(16).hex()


@server.before_request
def ensure_login():
    if session.get("user") is not None and session.get("boot_token") == server.config["BOOT_TOKEN"]:
        return None

    endpoint = request.endpoint or ""
    path = request.path or ""

    if endpoint in {"login.login", "login.login_static", "static"}:
        return None

    if path.startswith("/_dash") or path.startswith("/assets/") or path == "/favicon.ico":
        return None

    return redirect(url_for("login.login", next=request.url))


@server.route("/")
def root_redirect():
    return redirect(url_for("dash_redirect"))


@server.route("/dash")
def dash_redirect():
    return redirect("/dash/")


def _initialize_session_state() -> None:
    """Prepare default values for Streamlit session state."""

    if "experiments" not in st.session_state:
        st.session_state.experiments: List[Dict[str, Any]] = []

    if "experiment_selector" not in st.session_state:
        st.session_state.experiment_selector = "New Experiment"

    if "show_attachment_menu" not in st.session_state:
        st.session_state.show_attachment_menu = False

    if "pending_form_reset" not in st.session_state:
        st.session_state.pending_form_reset = False


def _toggle_attachment_menu() -> None:
    st.session_state.show_attachment_menu = not st.session_state.show_attachment_menu


def _reset_form() -> None:
    st.session_state.pending_form_reset = True
    st.session_state.show_attachment_menu = False


def _generate_experiment_name(
    start: Optional[date], end: Optional[date], count: int
) -> str:
    if start and end:
        return f"Experiment {count}: {start:%b %d} – {end:%b %d}"
    return f"Experiment {count}"


def _build_mock_results() -> Dict[str, Any]:
    timeline = pd.date_range("2025-04-01", "2025-04-15", freq="D")
    observed = [120, 122, 119, 125, 130, 128, 135, 140, 138, 142, 145, 147, 149, 150, 152]
    synthetic = [118, 119, 117, 118, 121, 123, 126, 128, 129, 131, 133, 134, 135, 136, 138]
    df = pd.DataFrame(
        {
            "Date": timeline,
            "Observed": observed,
            "Counterfactual": synthetic,
        }
    )

    lift_df = pd.DataFrame(
        {
            "Metric": ["Average Treatment Effect", "Credible Interval", "p-value"],
            "Value": ["+4.8%", "[+1.9%, +7.2%]", "0.013"],
        }
    )

    return {
        "series": df,
        "summary": {
            "Baseline": "128.5",
            "Observed": "135.4",
            "Lift": "+4.8%",
        },
        "lift_table": lift_df,
    }


def _render_top_bar(options: List[str]) -> None:
    col_logo, col_title, col_action = st.columns([1, 6, 2])
    with col_title:
        st.markdown(
            """
            <div class="top-title">New Experiment</div>
            """,
            unsafe_allow_html=True,
        )
    with col_action:
        if st.button("Reset", key="reset_top", type="primary"):
            st.session_state.experiment_selector = options[0]
            _reset_form()
            st.rerun()


def _render_sidebar() -> List[str]:
    st.sidebar.markdown("<div class='sidebar-title'>Experiments</div>", unsafe_allow_html=True)
    options = ["New Experiment"] + [exp["name"] for exp in st.session_state.experiments]
    if st.session_state.experiment_selector not in options:
        st.session_state.experiment_selector = options[0]

    st.sidebar.selectbox(
        "Experiment history",
        options=options,
        key="experiment_selector",
        label_visibility="collapsed",
    )
    return options


def _render_new_experiment_panel() -> None:
    st.markdown("<h2 class='hero-title'>What's on the agenda today?</h2>", unsafe_allow_html=True)
    container = st.container()
    with container:
        col_icon, col_input = st.columns([1, 12])
        with col_icon:
            st.button("＋", key="attachment_toggle", on_click=_toggle_attachment_menu)
        with col_input:
            st.text_area(
                "Ask anything",
                key="new_message",
                height=120,
                placeholder="Run an experiment from April 1, 2025 to April 15, 2025…",
                label_visibility="collapsed",
            )

        if st.session_state.show_attachment_menu:
            with st.container():
                st.markdown(
                    """
                    <div class="attachment-menu">
                        <div class="attachment-header">Add inputs directly</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                att_col1, att_col2 = st.columns(2)
                with att_col1:
                    st.date_input("Start date", key="start_date")
                    st.text_input(
                        "Covariates",
                        key="covariates",
                        placeholder="Channel spend, Seasonality…",
                    )
                with att_col2:
                    st.date_input("End date", key="end_date")
                    st.file_uploader(
                        "Dataset",
                        type=["csv", "xlsx"],
                        key="dataset_upload",
                        help="Attach a CSV or Excel file with your metric.",
                    )

    run_clicked = st.button("Run experiment", type="primary", use_container_width=True)

    if run_clicked:
        message = st.session_state.get("new_message", "").strip()
        start_date = st.session_state.get("start_date")
        end_date = st.session_state.get("end_date")
        covariates = st.session_state.get("covariates", "")
        dataset_upload = st.session_state.get("dataset_upload")

        name = _generate_experiment_name(start_date, end_date, len(st.session_state.experiments) + 1)

        experiment_data = {
            "name": name,
            "message": message or "Untitled experiment request",
            "start_date": start_date,
            "end_date": end_date,
            "covariates": covariates,
            "dataset_name": getattr(dataset_upload, "name", None),
            "results": _build_mock_results(),
        }

        st.session_state.experiments.append(experiment_data)
        st.session_state.pending_experiment_selector = name
        st.session_state.pending_form_reset = True
        st.rerun()


def _render_existing_experiment(experiment: Dict[str, Any]) -> None:
    st.markdown(f"<h2 class='hero-title'>{experiment['name']}</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("### Request summary")
        st.markdown(
            f"""
            <div class="message-card">
                <div class="message-title">Original brief</div>
                <div class="message-body">{experiment['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        meta_cols = st.columns(3)
        meta_cols[0].markdown(
            f"**Start date**<br>{experiment['start_date']:%b %d, %Y}", unsafe_allow_html=True
        )
        meta_cols[1].markdown(
            f"**End date**<br>{experiment['end_date']:%b %d, %Y}", unsafe_allow_html=True
        )
        dataset_label = experiment.get("dataset_name") or "No file attached"
        meta_cols[2].markdown(f"**Dataset**<br>{dataset_label}", unsafe_allow_html=True)

        st.markdown("---")

    summary = experiment["results"]["summary"]
    metric_cols = st.columns(len(summary))
    for idx, (label, value) in enumerate(summary.items()):
        metric_cols[idx].metric(label, value)

    series_df: pd.DataFrame = experiment["results"]["series"]
    chart = (
        alt.Chart(series_df)
        .transform_fold(["Observed", "Counterfactual"], as_=["Series", "Value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Metric"),
            color=alt.Color("Series:N", title=""),
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Effect overview")
    st.table(experiment["results"]["lift_table"])

    if experiment.get("covariates"):
        st.markdown("### Covariates")
        st.markdown(experiment["covariates"])


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        body {
            background-color: #f9fafb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #1f2933;
        }
        .top-title {
            font-size: 1.25rem;
            font-weight: 600;
            padding: 0.5rem 0;
        }
        .sidebar .sidebar-content {
            background-color: white;
        }
        .sidebar-title {
            font-size: 1rem;
            font-weight: 600;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        .attachment-menu {
            background: white;
            border: 1px solid #e4e7eb;
            border-radius: 12px;
            margin: 0.5rem 0 1rem;
            padding: 0.75rem 1rem;
        }
        .attachment-header {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .message-card {
            background: white;
            border-radius: 16px;
            padding: 1.25rem;
            border: 1px solid #e1e5ea;
            margin-bottom: 1.5rem;
        }
        .message-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .message-body {
            color: #334155;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Experimentation Console",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_styles()
    _initialize_session_state()

    pending_selector = st.session_state.pop("pending_experiment_selector", None)
    if pending_selector:
        st.session_state.experiment_selector = pending_selector

    if st.session_state.get("pending_form_reset"):
        st.session_state.pending_form_reset = False
        st.session_state.new_message = ""
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
        st.session_state.covariates = ""
        st.session_state.dataset_upload = None

    options = _render_sidebar()
    _render_top_bar(options)

    selected = st.session_state.get("experiment_selector", "New Experiment")

    if selected == "New Experiment":
        if "new_message" not in st.session_state:
            _reset_form()
        _render_new_experiment_panel()
    else:
        match = next(
            (exp for exp in st.session_state.experiments if exp["name"] == selected),
            None,
        )
        if match:
            _render_existing_experiment(match)
        else:
            st.info("Select an experiment from the sidebar or start a new one.")


if __name__ == "__main__":
    server.run()


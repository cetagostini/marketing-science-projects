"""Dash implementation of the experimentation console prototype."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

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
    no_update,
)
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from flask import session, has_request_context

from storage import load_experiments, save_experiments


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


def build_composer_chips(
    start: Optional[str],
    end: Optional[str],
    covariates: Optional[List[str]],
    dataset: Optional[str],
) -> List[html.Span]:
    chips: List[html.Span] = []
    if start and end:
        chips.append(html.Span(f"{start} → {end}", className="composer-chip"))
    for cov in covariates or []:
        chips.append(html.Span(cov, className="composer-chip"))
    if dataset:
        chips.append(html.Span(dataset, className="composer-chip"))
    return chips


def _generate_experiment_name(
    start: Optional[str], end: Optional[str], count: int
) -> str:
    if start and end:
        start_dt = datetime.fromisoformat(start).date()
        end_dt = datetime.fromisoformat(end).date()
        return f"Experiment {count}: {start_dt:%b %d} – {end_dt:%b %d}"
    return f"Experiment {count}"


def _build_mock_results() -> Dict[str, Any]:
    timeline = pd.date_range("2025-04-01", "2025-04-15", freq="D")
    observed = [120, 122, 119, 125, 130, 128, 135, 140, 138, 142, 145, 147, 149, 150, 152]
    synthetic = [118, 119, 117, 118, 121, 123, 126, 128, 129, 131, 133, 134, 135, 136, 138]
    df = pd.DataFrame({"Date": timeline, "Observed": observed, "Counterfactual": synthetic})

    lift_table = pd.DataFrame(
        {
            "Metric": ["Average Treatment Effect", "Credible Interval", "p-value"],
            "Value": ["+4.8%", "[+1.9%, +7.2%]", "0.013"],
        }
    )

    return {
        "series": df.to_dict("records"),
        "summary": {
            "Baseline": "128.5",
            "Observed": "135.4",
            "Lift": "+4.8%",
        },
        "lift_table": lift_table.to_dict("records"),
    }


def _default_form_values() -> Dict[str, Any]:
    today_iso = date.today().isoformat()
    return {
        "message": "",
        "start": today_iso,
        "end": today_iso,
        "covariates": [],
        "filename": None,
    }


def _build_series_chart(records: List[Dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(records)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Observed"],
            mode="lines+markers",
            name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Counterfactual"],
            mode="lines+markers",
            name="Counterfactual",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=24, t=32, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def handle_run_or_reset_logic(
    triggered: Optional[str],
    submit_clicks: Optional[int],
    reset_clicks: Optional[int],
    experiments: Optional[List[Dict[str, Any]]],
    message: str,
    start_date_value: Optional[str],
    end_date_value: Optional[str],
    covariates_value: Optional[str],
    upload_filename: Optional[str],
    **legacy_clicks: Any,
):
    experiments = experiments or []
    defaults = _default_form_values()

    if submit_clicks is None:
        submit_clicks = legacy_clicks.get("run_clicks")

    if triggered == "composer-send" and submit_clicks:
        start = start_date_value or defaults["start"]
        end = end_date_value or defaults["end"]
        name = _generate_experiment_name(start, end, len(experiments) + 1)
        experiment_data = {
            "name": name,
            "message": (message or "").strip() or "Untitled experiment request",
            "start_date": start,
            "end_date": end,
            "covariates": (covariates_value or "").strip(),
            "dataset_name": upload_filename,
            "results": _build_mock_results(),
        }
        updated_experiments = experiments + [experiment_data]
        return (
            updated_experiments,
            name,
            defaults["message"],
            defaults["start"],
            defaults["end"],
            defaults["covariates"],
            None,
            None,
        )

    if triggered == "reset-button" and (reset_clicks or 0) > 0:
        return (
            no_update,
            "New Experiment",
            defaults["message"],
            defaults["start"],
            defaults["end"],
            defaults["covariates"],
            None,
            None,
        )

    raise ValueError("Unhandled interaction")


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


def _render_experiment_detail(experiment: Dict[str, Any]) -> List[html.Component]:
    start_label = _format_date(experiment.get("start_date"))
    end_label = _format_date(experiment.get("end_date"))
    dataset_label = experiment.get("dataset_name") or "No file attached"

    summary_cards = [
        html.Div(
            [html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
            className="metric-card",
        )
        for label, value in experiment["results"]["summary"].items()
    ]

    chart = dcc.Graph(
        figure=_build_series_chart(experiment["results"]["series"]),
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

    covariate_section: List[html.Component]
    if experiment.get("covariates"):
        covariate_section = [
            html.H3("Covariates", className="section-title"),
            html.Div(experiment["covariates"], className="section-body"),
        ]
    else:
        covariate_section = []

    return [
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
                    ],
                    className="meta-grid",
                ),
            ],
            className="detail-card",
        ),
        html.Div(summary_cards, className="metric-grid"),
        chart,
        html.H3("Effect overview", className="section-title"),
        table,
    ] + covariate_section


def _format_date(value: Optional[str]) -> str:
    if not value:
        return "Not set"
    try:
        parsed = datetime.fromisoformat(value).date()
        return f"{parsed:%b %d, %Y}"
    except ValueError:
        return value


from app import server


app = Dash(__name__, server=server, url_base_pathname="/dash/")
app.title = "Experimentation Console"

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
            .experiment-card.active {
                border-color: #2563eb;
                background: #eff6ff;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
                transform: translateX(4px);
            }
            .experiment-card:hover {
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
    if not has_request_context():
        return []

    user_key = session.get("user", "default")
    experiments = session.get("experiments", {}).get(user_key)
    if experiments is None:
        experiments = load_experiments(user_key)
        session.setdefault("experiments", {})[user_key] = experiments
    return experiments or []


def serve_layout() -> html.Div:
    initial_experiments = _initial_experiments_state()
    return html.Div(
        [
            dcc.Store(id="experiments-store", data=initial_experiments),
            dcc.Store(id="sidebar-expanded", data=True),
            dcc.Store(id="selected-experiment", data="New Experiment"),
            dcc.Store(id="covariate-list", data=[]),
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
                                    html.Div("New Experiment", id="top-title", className="top-title"),
                                ],
                                className="top-bar",
                            ),
                            html.Div(
                                [
                            html.Div(
                                [
                                    html.H2("What are we testing today?", className="hero-title"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Button("＋", id="toggle-attachments", className="composer-icon"),
                                                    dcc.Textarea(
                                                        id="form-message",
                                                        placeholder="Describe the experiment you want to run…",
                                                        className="composer-input",
                                                        value="",
                                                    ),
                                                    html.Button("➤", id="composer-send", className="composer-icon send"),
                                                ],
                                                id="message-composer",
                                                className="message-composer",
                                            ),
                                            html.Div(id="composer-chips", className="composer-chip-row"),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Div("Date range", className="attachment-label"),
                                                            dcc.DatePickerRange(
                                                                id="date-range",
                                                                start_date=date.today().isoformat(),
                                                                end_date=date.today().isoformat(),
                                                                display_format="MMM D, YYYY",
                                                                className="date-picker",
                                                            ),
                                                            html.Div("Covariates", className="attachment-label"),
                                                            dcc.Input(
                                                                id="covariates-input",
                                                                type="text",
                                                                placeholder="Add a covariate and press Apply",
                                                            ),
                                                            html.Button("Add", id="add-covariate", n_clicks=0, className="composer-icon"),
                                                        ],
                                                        className="attachment-column",
                                                    ),
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
        ],
    )


app.layout = serve_layout


@app.callback(
    Output("experiment-list", "children"),
    Output("sidebar-expanded", "data"),
    Output("sidebar", "className"),
    Input("experiments-store", "data"),
    Input("selected-experiment", "data"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-expanded", "data"),
)
def update_sidebar(
    experiments: Optional[List[Dict[str, Any]]],
    selected: Optional[str],
    toggle_clicks: Optional[int],
    expanded_state: Optional[bool],
):
    experiments = experiments or []
    selected_value = selected or "New Experiment"

    cards = [
        html.Button(
            [
                html.Div("New Experiment", className="experiment-card-title"),
                html.Div("Start a fresh request", className="experiment-card-description"),
            ],
            className=f"experiment-card{' active' if selected_value == 'New Experiment' else ''}",
            id={"type": "experiment-card", "value": "New Experiment"},
            n_clicks=0,
        )
    ]

    for exp in experiments:
        subtitle = exp.get("message", "").strip() or "Untitled experiment request"
        cards.append(
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
    Input({"type": "experiment-card", "value": ALL}, "n_clicks"),
    State({"type": "experiment-card", "value": ALL}, "id"),
    State("selected-experiment", "data"),
    prevent_initial_call=True,
)
def select_experiment(cards_clicks, card_ids, current_selection):
    triggered = callback_context.triggered_id

    if not triggered:
        return current_selection

    if isinstance(triggered, dict):
        return triggered.get("value", current_selection)

    return current_selection


@app.callback(
    Output("top-title", "children"),
    Output("new-experiment-view", "style"),
    Output("experiment-detail-view", "style"),
    Output("experiment-detail-view", "children"),
    Input("selected-experiment", "data"),
    Input("experiments-store", "data"),
)
def update_main_view(selection: Optional[str], experiments: Optional[List[Dict[str, Any]]]):
    experiments = experiments or []
    selected_value = selection or "New Experiment"

    if selected_value != "New Experiment":
        match = next((exp for exp in experiments if exp["name"] == selection), None)
        if match:
            return (
                match["name"],
                {"display": "none"},
                {"display": "block"},
                _render_experiment_detail(match),
            )
    return (
        "New Experiment",
        {"display": "block"},
        {"display": "none"},
        [],
    )


@app.callback(
    Output("experiments-store", "data"),
    Output("selected-experiment", "data", allow_duplicate=True),
    Output("composer-chips", "children"),
    Output("attachment-area", "style", allow_duplicate=True),
    Output("attachment-area", "className", allow_duplicate=True),
    Output("form-message", "value", allow_duplicate=True),
    Output("date-range", "start_date", allow_duplicate=True),
    Output("date-range", "end_date", allow_duplicate=True),
    Output("covariates-input", "value", allow_duplicate=True),
    Output("upload-data", "contents", allow_duplicate=True),
    Output("upload-data", "filename", allow_duplicate=True),
    Input("composer-send", "n_clicks"),
    State("experiments-store", "data"),
    State("form-message", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("covariate-list", "data"),
    State("upload-data", "filename"),
    State("attachment-area", "className"),
    prevent_initial_call=True,
)
def handle_composer_actions(
    send_clicks: Optional[int],
    experiments: Optional[List[Dict[str, Any]]],
    message: str,
    start_date_value: Optional[str],
    end_date_value: Optional[str],
    covariates_list: Optional[List[str]],
    upload_filename: Optional[str],
    current_class: str,
):
    triggered = callback_context.triggered_id
    if triggered != "composer-send":
        raise PreventUpdate

    user_key = session.get("user", "default")
    experiments = experiments or []
    experiment_store = session.setdefault("experiments", {})
    if user_key not in experiment_store:
        experiment_store[user_key] = load_experiments(user_key)

    user_experiments = experiment_store[user_key]
    results = handle_run_or_reset_logic(
        triggered,
        send_clicks,
        None,
        user_experiments,
        message,
        start_date_value,
        end_date_value,
        ", ".join(covariates_list or []),
        upload_filename,
    )

    (
        experiments_state,
        selected_value,
        message_reset,
        start_reset,
        end_reset,
        covariates_reset,
        upload_contents_reset,
        upload_filename_reset,
    ) = results

    if experiments_state is not no_update:
        experiment_store[user_key] = experiments_state
        save_experiments(user_key, experiments_state)
    session.modified = True
    chips = build_composer_chips(start_reset, end_reset, covariates_list or [], upload_filename_reset)

    return (
        experiments_state,
        selected_value,
        chips,
        {"display": "none"},
        "attachment-popover attachment-top",
        message_reset,
        start_reset,
        end_reset,
        "",
        upload_contents_reset,
        upload_filename_reset,
    )


@app.callback(
    Output("covariate-list", "data"),
    Output("covariates-input", "value"),
    Input("add-covariate", "n_clicks"),
    State("covariate-list", "data"),
    State("covariates-input", "value"),
    prevent_initial_call=True,
)
def add_covariate(n_clicks: Optional[int], existing: Optional[List[str]], value: Optional[str]):
    if not n_clicks:
        return existing or [], value
    existing = existing or []
    new_value = (value or "").strip()
    if new_value and new_value not in existing:
        existing.append(new_value)
    return existing, ""


@app.callback(
    Output("attachment-area", "style"),
    Output("attachment-area", "className"),
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

    if style.get("display") == "none":
        return {"display": "none"}, "attachment-popover attachment-bottom"

    class_name = style.get(
        "className",
        current_class or "attachment-popover attachment-bottom",
    )
    return {"display": "block"}, class_name


if __name__ == "__main__":
    app.run(debug=True)



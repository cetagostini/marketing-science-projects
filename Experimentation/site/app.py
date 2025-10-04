"""Dash application for the experimentation site mock interface."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import dash
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    ALL,
    dcc,
    html,
    no_update,
)
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


def generate_placeholder_figure() -> go.Figure:
    """Create a static figure used for every experiment dashboard."""

    weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]
    actual = [120, 128, 134, 150, 163, 172]
    synthetic = [118, 120, 125, 129, 133, 137]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=actual,
            mode="lines+markers",
            name="Observed",
            line=dict(color="#6a5acd", width=3),
            marker=dict(size=8, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=synthetic,
            mode="lines+markers",
            name="Counterfactual",
            line=dict(color="#00bcd4", width=3, dash="dash"),
            marker=dict(size=8, symbol="square"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,18,34,0.85)",
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        xaxis_title="Timeline",
        yaxis_title="Performance Index",
        hovermode="x unified",
    )

    return fig


def build_dashboard(experiment: Dict[str, Optional[str]]) -> html.Div:
    """Assemble the dashboard contents for the active experiment."""

    figure = generate_placeholder_figure()
    metrics = [
        {"label": "Estimated Lift", "value": "+4.2%"},
        {"label": "Credible Interval", "value": "[+1.1%, +7.2%]"},
        {"label": "Posterior Probability", "value": "92%"},
    ]

    metric_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        html.Small(metric["label"], className="metric-label"),
                        html.H4(metric["value"], className="metric-value"),
                    ],
                    className="metric-card",
                ),
                md=4,
                sm=12,
            )
            for metric in metrics
        ],
        className="g-3",
    )

    attachments = html.Div(
        [
            html.Div(
                [
                    html.Span("Message", className="attachment-label"),
                    html.P(experiment.get("message") or "No details provided.", className="attachment-value"),
                ],
                className="attachment-item",
            ),
            html.Div(
                [
                    html.Span("Start", className="attachment-label"),
                    html.Span(experiment.get("start_date") or "—", className="attachment-value"),
                ],
                className="attachment-item",
            ),
            html.Div(
                [
                    html.Span("End", className="attachment-label"),
                    html.Span(experiment.get("end_date") or "—", className="attachment-value"),
                ],
                className="attachment-item",
            ),
            html.Div(
                [
                    html.Span("Dataset", className="attachment-label"),
                    html.Span(experiment.get("dataset_name") or "Not provided", className="attachment-value"),
                ],
                className="attachment-item",
            ),
            html.Div(
                [
                    html.Span("Covariates", className="attachment-label"),
                    html.Span(experiment.get("covariates") or "Not specified", className="attachment-value"),
                ],
                className="attachment-item",
            ),
        ],
        className="attachments-summary",
    )

    figure_section = dbc.Card(
        [
            dbc.CardHeader("Response vs. Synthetic Control", className="dashboard-card-header"),
            dbc.CardBody(dcc.Graph(figure=figure, config={"displayModeBar": False})),
        ],
        className="dashboard-card",
    )

    table_data = [
        {"metric": "Observed Impact", "value": "1.8"},
        {"metric": "Relative Uplift", "value": "4.2%"},
        {"metric": "Bayesian P(Impact > 0)", "value": "92%"},
        {"metric": "Inference Window", "value": "14 days"},
    ]

    summary_table = dbc.Card(
        [
            dbc.CardHeader("Impact Summary", className="dashboard-card-header"),
            dbc.CardBody(
                dash_table.DataTable(
                    data=table_data,
                    columns=[{"name": "Metric", "id": "metric"}, {"name": "Value", "id": "value"}],
                    style_as_list_view=True,
                    style_cell={"backgroundColor": "rgba(10, 10, 25, 0.7)", "color": "#f5f5f5", "border": "0"},
                    style_header={"display": "none"},
                    style_data_conditional=[
                        {
                            "if": {"row_index": "even"},
                            "backgroundColor": "rgba(20, 20, 45, 0.85)",
                        }
                    ],
                )
            ),
        ],
        className="dashboard-card",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H3(f"Experiment Insights — {experiment['name']}", className="dashboard-title"),
                    html.P(
                        "These are placeholder analytics. The causal impact engine will be wired in future iterations.",
                        className="dashboard-description",
                    ),
                ],
                className="dashboard-header",
            ),
            attachments,
            metric_cards,
            dbc.Row([
                dbc.Col(figure_section, lg=8, md=12, className="mb-3"),
                dbc.Col(summary_table, lg=4, md=12, className="mb-3"),
            ]),
        ],
        className="dashboard-surface",
    )


def build_placeholder_dashboard() -> html.Div:
    """Return instructions when no experiment is active."""

    return html.Div(
        [
            html.H3("Run your first experiment", className="dashboard-title"),
            html.P(
                "Describe the experimental request or specify the period, geography, and dataset, then press Run Experiment.",
                className="dashboard-description",
            ),
            html.Ul(
                [
                    html.Li("Use the + button to capture structured details like dates and covariates."),
                    html.Li("Upload a CSV to serve as the metric source."),
                    html.Li("Every experiment you run will appear in the sidebar list for easy recall."),
                ],
                className="dashboard-instructions",
            ),
        ],
        className="dashboard-placeholder",
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Experimentation Site",
)
server = app.server


app.layout = html.Div(
    className="app-surface",
    children=[
        dcc.Store(id="session-experiments", storage_type="session", data={"experiments": [], "active_id": None}),
        dcc.Store(id="ui-state", storage_type="session", data={"sidebar_collapsed": False}),
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "☰",
                                    id="sidebar-toggle",
                                    color="link",
                                    class_name="sidebar-toggle-btn",
                                ),
                                width="auto",
                            ),
                            dbc.Col(dbc.NavbarBrand("Experimentation Site", class_name="navbar-brand")),
                        ],
                        align="center",
                        class_name="g-3",
                        justify="between",
                    )
                ],
                fluid=True,
            ),
            class_name="top-navbar",
            color="dark",
            dark=True,
        ),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div("Experiments", className="sidebar-title"),
                                dbc.Button(
                                    "+ New Experiment",
                                    id="sidebar-new-experiment",
                                    color="secondary",
                                    class_name="sidebar-new-btn",
                                ),
                                dcc.Dropdown(
                                    id="experiment-selector",
                                    options=[],
                                    placeholder="Jump to a previous experiment",
                                    className="experiment-dropdown",
                                ),
                                html.Div(id="experiment-summary-list", className="experiment-summary"),
                            ],
                            className="sidebar-card",
                        ),
                        lg=3,
                        md=4,
                        sm=12,
                        id="sidebar-column",
                        class_name="sidebar-column expanded",
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                html.Div(id="experiment-header", className="content-title", children="New Experiment"),
                                html.P(
                                    "Summarize your experiment request below. You can paste plain language instructions and attach structured inputs.",
                                    className="content-subtitle",
                                ),
                                dcc.Textarea(
                                    id="experiment-input",
                                    placeholder="Describe the experiment you want to run...",
                                    className="prompt-textarea",
                                    rows=6,
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            [html.Span("+", className="attach-icon"), html.Span(" Attach details")],
                                            id="toggle-attachments",
                                            class_name="attach-btn",
                                            color="link",
                                        ),
                                        dbc.Collapse(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.DatePickerSingle(
                                                            id="start-date",
                                                            placeholder="Start date",
                                                            display_format="MMM DD, YYYY",
                                                            className="date-picker",
                                                        ),
                                                        md=6,
                                                        sm=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.DatePickerSingle(
                                                            id="end-date",
                                                            placeholder="End date",
                                                            display_format="MMM DD, YYYY",
                                                            className="date-picker",
                                                        ),
                                                        md=6,
                                                        sm=12,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Upload(
                                                            id="upload-dataset",
                                                            children=html.Div(
                                                                [
                                                                    html.Span("Drag & Drop or ", className="upload-text"),
                                                                    html.Button("Select File", className="upload-btn"),
                                                                ]
                                                            ),
                                                            className="upload-container",
                                                            multiple=False,
                                                        ),
                                                        md=6,
                                                        sm=12,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            id="covariates",
                                                            placeholder="Comma-separated covariates",
                                                            class_name="covariate-input",
                                                        ),
                                                        md=6,
                                                        sm=12,
                                                    ),
                                                ],
                                                class_name="g-3 attachments-grid",
                                            ),
                                            id="attachments-panel",
                                            is_open=False,
                                        ),
                                        html.Div(id="dataset-name-display", className="dataset-display"),
                                    ],
                                    className="attachment-section",
                                ),
                                dbc.Button(
                                    "Run Experiment",
                                    id="run-experiment-btn",
                                    color="success",
                                    class_name="run-btn",
                                ),
                                html.Div(id="dashboard-output", className="dashboard-output"),
                            ],
                            className="content-card",
                        ),
                        lg=9,
                        md=8,
                        sm=12,
                        id="content-column",
                        class_name="content-column",
                    ),
                ],
                class_name="g-4 main-grid",
            ),
            fluid=True,
            class_name="main-container",
        ),
    ],
)


@app.callback(
    Output("attachments-panel", "is_open"),
    Input("toggle-attachments", "n_clicks"),
    Input("sidebar-new-experiment", "n_clicks"),
    State("attachments-panel", "is_open"),
    prevent_initial_call=True,
)
def toggle_attachments(
    toggle_clicks: Optional[int], sidebar_clicks: Optional[int], is_open: bool
) -> bool:
    """Toggle attachment panel or close it when starting a new experiment."""

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered_id
    if trigger == "toggle-attachments":
        return not is_open

    return False


@app.callback(
    Output("session-experiments", "data"),
    Output("experiment-selector", "options"),
    Output("experiment-selector", "value"),
    Input("run-experiment-btn", "n_clicks"),
    Input("experiment-selector", "value"),
    Input("sidebar-new-experiment", "n_clicks"),
    Input({"type": "experiment-card", "index": ALL}, "n_clicks"),
    State("session-experiments", "data"),
    State("experiment-input", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("covariates", "value"),
    State("upload-dataset", "filename"),
    prevent_initial_call=True,
)
def manage_experiments(
    run_clicks: Optional[int],
    selected_experiment: Optional[str],
    sidebar_new_clicks: Optional[int],
    card_clicks: List[Optional[int]],
    store: Optional[Dict[str, List[Dict[str, Optional[str]]]]],
    message: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    covariates: Optional[str],
    dataset_filename: Optional[str | List[str]],
):
    """Handle experiment creation, selection, and reset events."""

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered_id

    store = store or {"experiments": [], "active_id": None}
    experiments = store.get("experiments", [])

    def build_options(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [{"label": item["name"], "value": item["id"]} for item in items]

    if trigger == "run-experiment-btn":
        if not run_clicks:
            raise PreventUpdate

        dataset_name: Optional[str]
        if isinstance(dataset_filename, list):
            dataset_name = ", ".join(dataset_filename)
        else:
            dataset_name = dataset_filename

        new_id = str(uuid.uuid4())
        experiment_name = f"Experiment {len(experiments) + 1}"
        created_at = datetime.utcnow().strftime("%b %d, %Y %H:%M UTC")

        new_experiment = {
            "id": new_id,
            "name": experiment_name,
            "message": (message or "").strip() or "No details provided.",
            "start_date": start_date,
            "end_date": end_date,
            "covariates": (covariates or "").strip(),
            "dataset_name": dataset_name,
            "created_at": created_at,
        }

        new_experiments = experiments + [new_experiment]
        new_store = {"experiments": new_experiments, "active_id": new_id}
        options = build_options(new_experiments)
        return new_store, options, new_id

    if trigger == "experiment-selector":
        options = build_options(experiments)
        updated_store = {"experiments": experiments, "active_id": selected_experiment}
        return updated_store, options, selected_experiment

    if trigger == "sidebar-new-experiment":
        options = build_options(experiments)
        updated_store = {"experiments": experiments, "active_id": None}
        return updated_store, options, None

    if isinstance(trigger, dict) and trigger.get("type") == "experiment-card":
        experiment_id = trigger.get("index")
        options = build_options(experiments)
        updated_store = {"experiments": experiments, "active_id": experiment_id}
        return updated_store, options, experiment_id

    raise PreventUpdate


@app.callback(
    Output("experiment-input", "value"),
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("covariates", "value"),
    Output("dashboard-output", "children"),
    Output("experiment-header", "children"),
    Output("dataset-name-display", "children"),
    Output("experiment-summary-list", "children"),
    Output("sidebar-column", "className"),
    Output("content-column", "className"),
    Output("ui-state", "data"),
    Input("session-experiments", "data"),
    Input("upload-dataset", "filename"),
    Input("sidebar-toggle", "n_clicks"),
    State("ui-state", "data"),
)
def populate_experiment_view(
    store: Optional[Dict[str, List[Dict[str, Optional[str]]]]],
    uploaded_filename: Optional[str | List[str]],
    sidebar_toggle_clicks: Optional[int],
    ui_state: Optional[Dict[str, bool]],
):
    """Update form fields, dashboard, and sidebar summaries."""

    ctx = dash.callback_context
    triggered = ctx.triggered_id if ctx.triggered else None

    store = store or {"experiments": [], "active_id": None}
    experiments: List[Dict[str, Optional[str]]] = store.get("experiments", [])
    active_id: Optional[str] = store.get("active_id")
    ui_state = ui_state or {"sidebar_collapsed": False}
    sidebar_state = not ui_state.get("sidebar_collapsed", False)

    summary_cards = [
        dbc.Button(
            [
                html.Span(exp["name"], className="summary-name"),
                html.Span(exp.get("created_at", ""), className="summary-date"),
            ],
            id={"type": "experiment-card", "index": exp["id"]},
            class_name="summary-item active" if exp["id"] == active_id else "summary-item",
            color="link",
        )
        for exp in experiments
    ]

    def compose_classes(is_sidebar_open: bool) -> tuple[str, str]:
        if is_sidebar_open:
            return "sidebar-column expanded", "content-column"
        return "sidebar-column collapsed", "content-column expanded"

    if triggered == "upload-dataset":
        if uploaded_filename:
            name = ", ".join(uploaded_filename) if isinstance(uploaded_filename, list) else uploaded_filename
            dataset_display = f"Attached: {name}"
        else:
            dataset_display = "No dataset attached"

        sidebar_class, content_class = compose_classes(sidebar_state)

        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dataset_display,
            summary_cards,
            sidebar_state,
            sidebar_class,
            content_class,
            ui_state,
        )

    if triggered == "sidebar-toggle":
        toggled_state = not sidebar_state
        updated_ui_state = {"sidebar_collapsed": not toggled_state}
        sidebar_class, content_class = compose_classes(toggled_state)
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            summary_cards,
            sidebar_class,
            content_class,
            updated_ui_state,
        )

    if not experiments or not active_id:
        sidebar_class, content_class = compose_classes(sidebar_state)
        return (
            "",
            None,
            None,
            "",
            build_placeholder_dashboard(),
            "New Experiment",
            "No dataset attached",
            summary_cards,
            sidebar_class,
            content_class,
            ui_state,
        )

    active_experiment = next((exp for exp in experiments if exp["id"] == active_id), None)

    if not active_experiment:
        sidebar_class, content_class = compose_classes(sidebar_state)
        return (
            "",
            None,
            None,
            "",
            build_placeholder_dashboard(),
            "New Experiment",
            "No dataset attached",
            summary_cards,
            sidebar_class,
            content_class,
            ui_state,
        )

    dataset_label = (
        f"Attached: {active_experiment['dataset_name']}"
        if active_experiment.get("dataset_name")
        else "No dataset attached"
    )

    sidebar_class, content_class = compose_classes(sidebar_state)

    return (
        active_experiment.get("message"),
        active_experiment.get("start_date"),
        active_experiment.get("end_date"),
        active_experiment.get("covariates"),
        build_dashboard(active_experiment),
        f"Viewing {active_experiment['name']}",
        dataset_label,
        summary_cards,
        sidebar_class,
        content_class,
        ui_state,
    )


if __name__ == "__main__":
    app.run(debug=True)


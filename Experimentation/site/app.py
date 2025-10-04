import datetime
import uuid
from typing import Any, Dict, List, Optional

import dash
from dash import Dash, Input, Output, State, callback_context, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


APP_TITLE = "Experimentation Studio"
DEFAULT_DASHBOARD_DATA = pd.DataFrame(
    {
        "Date": pd.date_range("2025-03-15", periods=14, freq="D"),
        "Control": [120, 118, 122, 125, 130, 128, 132, 135, 138, 140, 142, 144, 146, 148],
        "Treatment": [119, 120, 126, 131, 137, 139, 141, 145, 150, 153, 155, 158, 160, 164],
    }
)
DEFAULT_LIFT_SUMMARY = pd.DataFrame(
    {
        "Metric": ["Average Uplift", "Peak Uplift", "Post-Period Uplift"],
        "Value": [5.4, 12.8, 8.9],
    }
)


def _make_experiment_name(counter: int, message: Optional[str]) -> str:
    if message:
        preview = " ".join(message.split())[:24].rstrip()
        if preview:
            return f"Experiment {counter}: {preview}"
    return f"Experiment {counter}"


def create_dashboard_components(experiment: Dict[str, Any]) -> html.Div:
    fig = px.line(
        DEFAULT_DASHBOARD_DATA,
        x="Date",
        y=["Control", "Treatment"],
        labels={"value": "Metric", "variable": "Group"},
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="rgba(12,14,30,0.8)",
        plot_bgcolor="rgba(12,14,30,0.5)",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        title=dict(
            text="Estimated Effect Over Time",
            font=dict(color="#B8B4FF", size=20),
            x=0.5,
        ),
    )

    lift_fig = px.bar(
        DEFAULT_LIFT_SUMMARY,
        x="Metric",
        y="Value",
        labels={"Value": "Lift (%)"},
        template="plotly_dark",
        color="Metric",
        color_discrete_sequence=["#7F5AF0", "#2CB1FF", "#00B4D8"],
    )
    lift_fig.update_layout(
        margin=dict(l=0, r=0, t=32, b=0),
        paper_bgcolor="rgba(12,14,30,0.8)",
        plot_bgcolor="rgba(12,14,30,0.5)",
        title=dict(
            text="Lift Summary",
            font=dict(color="#B8B4FF", size=20),
            x=0.5,
        ),
        xaxis=dict(color="#E0E7FF"),
        yaxis=dict(color="#E0E7FF"),
    )

    info_cards = html.Div(
        className="dashboard-cards",
        children=[
            html.Div(
                className="dashboard-card",
                children=[
                    html.Div("Primary Metric", className="card-label"),
                    html.Div("Conversion Rate", className="card-value"),
                ],
            ),
            html.Div(
                className="dashboard-card",
                children=[
                    html.Div("Start → End", className="card-label"),
                    html.Div(
                        f"{experiment.get('start_date', 'Apr 01, 2025')} → {experiment.get('end_date', 'Apr 15, 2025')}",
                        className="card-value",
                    ),
                ],
            ),
            html.Div(
                className="dashboard-card",
                children=[
                    html.Div("Region", className="card-label"),
                    html.Div(experiment.get("dataset", "Germany Retail"), className="card-value"),
                ],
            ),
            html.Div(
                className="dashboard-card",
                children=[
                    html.Div("Status", className="card-label"),
                    html.Div("Completed", className="card-value status"),
                ],
            ),
        ],
    )

    figure_row = html.Div(
        className="figure-row",
        children=[
            html.Div(className="figure-card", children=[dcc.Graph(figure=fig, config={"displayModeBar": False})]),
            html.Div(className="figure-card", children=[dcc.Graph(figure=lift_fig, config={"displayModeBar": False})]),
        ],
    )

    notes = html.Div(
        className="insights-panel",
        children=[
            html.H3("Insights", className="panel-title"),
            html.Ul(
                children=[
                    html.Li("Treatment outperforms control by ~8% post intervention."),
                    html.Li("No significant pre-period imbalance detected."),
                    html.Li("Covariates remain stable; uplift is likely attributable to the experiment."),
                ]
            ),
        ],
    )

    experiment_brief = html.Div(
        className="experiment-brief",
        children=[
            html.H3("Experiment Summary", className="panel-title"),
            html.Div(
                className="brief-item",
                children=[
                    html.Span("Prompt", className="brief-label"),
                    html.P(experiment.get("message", "Not provided")),
                ],
            ),
            html.Div(
                className="brief-item",
                children=[
                    html.Span("Covariates", className="brief-label"),
                    html.P(experiment.get("covariates", "Seasonality, Spend")),
                ],
            ),
        ],
    )

    return html.Div(
        className="dashboard-wrapper",
        children=[info_cards, figure_row, html.Div(className="panel-grid", children=[notes, experiment_brief])],
    )


app: Dash = dash.Dash(__name__, external_stylesheets=[dbc.icons.BOOTSTRAP])
app.title = APP_TITLE

app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Store(id="experiments-store", data={"experiments": [], "counter": 0, "selected_id": None}),
        dcc.Store(id="attachments-store", data={"open": False}),
        html.Div(
            className="top-bar",
            children=[
                html.Div(APP_TITLE, className="app-title"),
                html.Button("New Experiment", id="new-experiment-button", className="primary-button"),
            ],
        ),
        html.Div(
            className="workspace",
            children=[
                html.Div(
                    className="sidebar",
                    children=[
                        html.Div("Your Experiments", className="sidebar-title"),
                        dcc.Dropdown(
                            id="experiment-selector",
                            placeholder="Jump to an experiment",
                            className="experiment-dropdown",
                            clearable=False,
                        ),
                        html.Div(id="experiment-list", className="experiment-list"),
                    ],
                ),
                html.Div(
                    className="main-content",
                    children=[
                        html.Div(
                            className="prompt-card",
                            children=[
                                html.Div("What are you researching?", className="prompt-title"),
                                html.Div(
                                    className="message-box",
                                    children=[
                                        dcc.Textarea(
                                            id="message-input",
                                            placeholder=(
                                                "Run an experiment from April 1, 2025 to April 15, 2025, in Germany.\n"
                                                "Here is a dataset with my metric in Germany—check if the experiment was successful."
                                            ),
                                            className="message-textarea",
                                            value="",
                                        ),
                                        html.Div(
                                            className="attachment-controls",
                                            children=[
                                                html.Button("+", id="attachment-toggle", className="attachment-button"),
                                                html.Span("Add details", className="attachment-label"),
                                            ],
                                        ),
                                        html.Div(
                                            id="attachment-panel",
                                            className="attachment-panel hidden",
                                            children=[
                                                html.Div(
                                                    className="attachment-field",
                                                    children=[
                                                html.Span("Start date", className="field-label"),
                                                dcc.DatePickerSingle(
                                                    id="start-date",
                                                    className="field-input date-input",
                                                    display_format="YYYY-MM-DD",
                                                    placeholder="YYYY-MM-DD",
                                                ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="attachment-field",
                                                    children=[
                                                        html.Span("End date", className="field-label"),
                                                dcc.DatePickerSingle(
                                                    id="end-date",
                                                    className="field-input date-input",
                                                    display_format="YYYY-MM-DD",
                                                    placeholder="YYYY-MM-DD",
                                                ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="attachment-field",
                                                    children=[
                                                        html.Span("Dataset", className="field-label"),
                                                        dcc.Input(
                                                            id="dataset-input",
                                                            type="text",
                                                            placeholder="Upload path or table name",
                                                            className="field-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="attachment-field",
                                                    children=[
                                                        html.Span("Covariates", className="field-label"),
                                                        dcc.Input(
                                                            id="covariates-input",
                                                            type="text",
                                                            placeholder="Seasonality, Spend, CPC",
                                                            className="field-input",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Button("Run mock experiment", id="run-experiment-button", className="primary-button"),
                            ],
                        ),
                        html.Div(id="dashboard-container", className="dashboard-container"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("attachments-store", "data"),
    Input("attachment-toggle", "n_clicks"),
    State("attachments-store", "data"),
    prevent_initial_call=True,
)
def toggle_attachments(n_clicks: int, state: Dict[str, Any]) -> Dict[str, Any]:
    open_state = state.get("open", False)
    return {"open": not open_state}


@app.callback(
    Output("attachment-panel", "className"),
    Input("attachments-store", "data"),
)
def set_attachment_panel(data: Dict[str, Any]) -> str:
    base_class = "attachment-panel"
    if not data.get("open"):
        return f"{base_class} hidden"
    return base_class


@app.callback(
    Output("experiment-selector", "options"),
    Output("experiment-list", "children"),
    Input("experiments-store", "data"),
)
def refresh_experiment_list(store_data: Dict[str, Any]):
    experiments: List[Dict[str, Any]] = store_data.get("experiments", [])
    options = [
        {"label": exp["name"], "value": exp["id"]}
        for exp in experiments
    ]

    cards = []
    for exp in experiments:
        cards.append(
            html.Div(
                className="sidebar-experiment",
                children=[
                    html.Div(exp["name"], className="sidebar-experiment-name"),
                    html.Div(
                        f"{exp.get('start_date', 'Start?')} → {exp.get('end_date', 'End?')}",
                        className="sidebar-experiment-meta",
                    ),
                ],
            )
        )

    empty_state = html.Div("No experiments yet. Run one to get started!", className="sidebar-empty")
    return options, cards if cards else empty_state


@app.callback(
    Output("experiments-store", "data"),
    Output("experiment-selector", "value"),
    Output("message-input", "value"),
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("dataset-input", "value"),
    Output("covariates-input", "value"),
    Input("run-experiment-button", "n_clicks"),
    Input("new-experiment-button", "n_clicks"),
    State("message-input", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("dataset-input", "value"),
    State("covariates-input", "value"),
    State("experiments-store", "data"),
    prevent_initial_call=True,
)
def handle_experiment_actions(
    run_clicks: Optional[int],
    new_clicks: Optional[int],
    message: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    dataset: Optional[str],
    covariates: Optional[str],
    store_data: Dict[str, Any],
):
    triggered = callback_context.triggered_id
    experiments = store_data.get("experiments", [])
    counter = store_data.get("counter", 0)

    if triggered == "run-experiment-button":
        if not message and not any([start_date, end_date, dataset, covariates]):
            return (
                store_data,
                store_data.get("selected_id"),
                message,
                start_date,
                end_date,
                dataset,
                covariates,
            )

        counter += 1
        experiment_id = str(uuid.uuid4())
        name = _make_experiment_name(counter, message)
        experiment = {
            "id": experiment_id,
            "name": name,
            "message": message or "",
            "start_date": start_date or "2025-04-01",
            "end_date": end_date or "2025-04-15",
            "dataset": dataset or "Germany Retail",
            "covariates": covariates or "Seasonality, Spend",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        experiments.append(experiment)
        store_data = {"experiments": experiments, "counter": counter, "selected_id": experiment_id}
        return store_data, experiment_id, "", None, None, "", ""

    if triggered == "new-experiment-button":
        store_data["selected_id"] = None
        return store_data, None, "", None, None, "", ""

    return store_data, store_data.get("selected_id"), message, start_date, end_date, dataset, covariates


@app.callback(
    Output("dashboard-container", "children"),
    Input("experiment-selector", "value"),
    Input("experiments-store", "data"),
)
def render_dashboard(selected_id: Optional[str], store_data: Dict[str, Any]):
    experiments = store_data.get("experiments", [])
    experiment_lookup = {exp["id"]: exp for exp in experiments}

    if selected_id and selected_id in experiment_lookup:
        experiment = experiment_lookup[selected_id]
        return create_dashboard_components(experiment)

    return html.Div(
        className="dashboard-empty",
        children=[
            html.H2("Ready when you are", className="empty-title"),
            html.P(
                "Draft a prompt or add structured inputs, then run a mock experiment to see insights here.",
                className="empty-description",
            ),
        ],
    )


if __name__ == "__main__":
    app.run(debug=True)

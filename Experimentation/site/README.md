# Experimentation Studio (Mock Interface)

This project provides a mock experimentation workspace built with [Dash](https://dash.plotly.com/) and Plotly. It is designed to showcase the proposed workflow for configuring marketing science experiments before connecting to real CausalPy runs.

## Features

- **Prompt-driven experiments**: Capture free-form experiment requests and structured metadata (start/end dates, dataset, covariates).
- **Session-aware navigation**: Saved experiments persist during the browser session and can be revisited from the sidebar dropdown.
- **Stylized dashboard**: Each mock experiment produces the same sample dashboard to demonstrate the future presentation layer for causal analysis results.

## Getting Started

### 1. Create the Conda environment

```bash
conda env create -f env.yml
conda activate experimentation_app
```

The environment installs Python 3.12, CausalPy, and supporting Dash/Plotly packages required to run the mock interface locally.

### 2. Launch the Dash application

```bash
python app.py
```

The development server defaults to `http://127.0.0.1:8050/`.

## Using the Interface

1. **Draft a request**: Enter an experiment description in the message area.
2. **Add structured inputs**: Click the `+` icon to toggle fields for start/end date, dataset, and covariates.
3. **Run the mock experiment**: Press **Run mock experiment** to save the request. A prebuilt dashboard appears with illustrative charts and insights.
4. **Switch between experiments**: Use the sidebar dropdown to jump back to previous mock experiments created during the current session.
5. **Start fresh**: Select **New Experiment** in the top bar to clear the form and prepare a new request while keeping saved experiments available in the sidebar.

## Next Steps

This scaffold can be extended by integrating real data ingestion, invoking CausalPy estimators, and persisting experiments to a database or API. The current version focuses on the look-and-feel of the eventual experimentation studio.

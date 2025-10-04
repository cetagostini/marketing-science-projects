# Experimentation Site (Mock)

This Plotly Dash application delivers a mock interface for orchestrating marketing experiments that will eventually leverage CausalPy.

## Features

- Modern interface inspired by research assistant tooling.
- Session-based experiment management and quick navigation.
- Rich input form with attachments for structured metadata.
- Placeholder dashboard with synthetic results (figure + summary cards).

## Prerequisites

- `conda` (Miniconda or Anaconda)
- macOS / Linux / Windows with Python 3.12 compatible toolchain

## Setup

1. **Clone the repository (or pull latest changes)**

   ```bash
   git clone <repo-url>
   cd marketing-science-projects/Experimentation/site
   ```

2. **Create and activate Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate experimentation_app
   ```

3. **Install optional local dependencies**

   If you prefer using `pip` inside the environment, ensure you have latest `pip`:

   ```bash
   pip install --upgrade pip
   ```

## Run the App

```bash
python app.py
```

The development server starts at `http://127.0.0.1:8050/` by default. The app runs in Dash debug mode for hot reloading.

## Using the Interface

1. **Compose Request**: Enter a natural language description in the prompt area.
2. **Attach Details**: Click `+ Attach details` to add structured metadata (start/end dates, dataset, covariates).
3. **Run Experiment**: Press `Run Experiment` to save the experiment. A mock dashboard with placeholder insights renders.
4. **Navigate Sessions**: Use the sidebar dropdown or cards to revisit previous experiments in the current browser session.
5. **Start New**: Use the top bar or sidebar `New Experiment` buttons to reset the form for the next experiment.

## Notes & Next Steps

- Current dashboard results are static. Integrations with CausalPy will populate real analyses later.
- Experiments persist only for the active browser session (Dash `dcc.Store`).
- The `data/` directory is reserved for future local uploads or sample datasets.
- Styling lives under `assets/styles.css`; adjust palette or layout there.

## Troubleshooting

- If the Dash server fails to start, confirm the Conda environment activated and `dash`, `plotly`, `dash-bootstrap-components` installed.
- For websocket reload issues, try visiting via `http://localhost:8050` instead of `127.0.0.1`.

## License

Refer to the root project license for overall terms.



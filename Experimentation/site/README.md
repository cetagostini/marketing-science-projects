# Experimentation Console (Mock Interface)

This folder contains a Streamlit prototype that mimics an experimentation workspace.  
It lets analysts describe an experiment request, attach structured inputs (start date, end date, dataset, and covariates), and switch between previously generated dashboards that are stored in the active Streamlit session.

## Features

- **Inbox-style homepage** with a primary text box inspired by ChatGPT's interface.
- **Attachment menu** with quick inputs for experiment metadata and dataset upload.
- **Session persistence** – every experiment submission is saved for the current browser session and listed in the sidebar.
- **Mock dashboard** – after any submission, a static causal impact view (metrics, chart, and lift table) is displayed.
- **Quick reset** – use the _New Experiment_ navigation to clear the form and start again.

## Requirements

Create and activate a Python 3.11 environment. You can either use `env.yml` with Conda or `requirements.txt` with `pip`.

### Option 1 – Conda (recommended)

```bash
conda env create -f env.yml
conda activate experimentation-site
```

### Option 2 – virtual environment with pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

Inside this `Experimentation/site` directory execute:

```bash
streamlit run app.py
```

The interface opens in your browser (default http://localhost:8501). Create experiments by typing a prompt, optionally adding attachments through the **＋** button, and pressing **Run experiment**. Each submission generates the same mock dashboard and is added to the sidebar so you can revisit it during the session.

## Notes

- Dataset uploads are optional and stored only in memory for the active session.
- The dashboard content is static and does not execute real causal inference yet; it is a placeholder for future CausalPy integration.

# Experimentation Site

Mock interface for orchestrating marketing experiments with CausalPy, built with Plotly Dash.

## Project Layout

```
Experimentation/
└── site/
    ├── app.py
    ├── assets/
    │   └── styles.css
    ├── data/
    ├── environment.yml
    └── README.md
```

## Quickstart

1. **Clone & Navigate**

   ```bash
   git clone <repo-url>
   cd marketing-science-projects/Experimentation/site
   ```

2. **Create Environment**

   ```bash
   conda env create -f environment.yml
   conda activate experimentation_app
   ```

3. **Run the Mock App**

   ```bash
   python app.py
   ```

## Current Functionality

- Compose experiment instructions in natural language
- Attach explicit metadata (dates, covariates, dataset)
- Save multiple experiments per session and switch between them via sidebar
- Display a reusable placeholder dashboard for each experiment

## Roadmap

- Hook up real CausalPy workflows
- Persist experiments across sessions
- Integrate data ingestion & validation
- Expand dashboard visuals and statistical summaries


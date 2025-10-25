# Quick Start: Running Fixed Experimentation App

## What Was Fixed

The main issue preventing tabs from being created was **an invalid OpenAI model name** (`"gpt-4.1"` → `"gpt-4o"`). This caused all experiment runs to fail silently.

Additional improvements:
- ✅ Added comprehensive logging throughout the pipeline
- ✅ Added user-visible error messages
- ✅ Improved error handling in callbacks
- ✅ Fixed deprecated test

## How to Run

### 1. Activate the Conda Environment

```bash
conda activate synth_experiments
```

### 2. Set Your OpenAI API Key

Make sure you have your OpenAI API key set:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to your `.env` file in `Experimentation/site/`:

```
OPENAI_API_KEY=your-api-key-here
APP_SECRET=your-secret-key
APP_USERS=alice:password
```

### 3. Run the App

Navigate to the site directory:

```bash
cd /Users/carlostrujillo/Documents/GitHub/marketing-science-projects/Experimentation/site
```

Run the server:

```bash
python -m gunicorn server:server
```

Or for development with auto-reload:

```bash
python -m gunicorn server:server --reload
```

### 4. Access the App

Open your browser and go to:
```
http://127.0.0.1:8000/
```

Login with credentials from `APP_USERS` (default: `alice:password`)

## How to Test

### Test 1: Upload a CSV and Run an Experiment

1. Click the **+** button to show the attachment area
2. Upload a CSV file with time series data (must have a date column and metric columns)
3. Enter a message like:
   ```
   I want to run an experiment from March 1, 2025 to March 20, 2025.
   The target variable is DEU. 
   The control countries are SWE, NOR, DNK, FIN, ISL, and GBR.
   ```
4. Click the send button (➤)

### Expected Behavior:

**Watch the Terminal Logs** for:
```
=== handle_composer_actions called ===
Triggered by: composer-send
Message: I want to run an experiment...
=== run_experiment_pipeline called ===
=== extract_experiment_details called ===
Using model: gpt-4o
LLM response received
=== extract_date_column called ===
=== run_experiment called ===
Experiment object created, running model...
Model run completed
Experiment saved! Total experiments now: 1
```

**In the UI**:
- A new tab should appear in the left sidebar with the experiment name
- The view should switch to show the experiment results
- You should see metrics, a chart, and an effect overview table

### Test 2: Error Handling

Try submitting without a dataset - you should see:
```
⚠ Please attach a dataset before submitting.
```

## Viewing Logs

All operations are now logged. To see detailed logs:

```bash
# Run with logging visible
python -m gunicorn server:server --log-level=info
```

You'll see detailed traces of:
- When callbacks are triggered
- What data is being processed
- LLM extraction results
- Model training progress
- Any errors that occur

## Troubleshooting

### "OpenAI client is not configured"
- Set the `OPENAI_API_KEY` environment variable
- Restart the server after setting the key

### "LLM suggested 'X' which is not present in dataframe columns"
- The LLM couldn't identify the date column correctly
- Make sure your CSV has a clear date/time column

### Tabs Still Not Appearing
- Check the terminal logs for errors
- Verify your OpenAI API key is valid
- Ensure your CSV has the required columns
- Check that the LLM successfully extracted experiment parameters

### See More Details
Read `DEBUG_FINDINGS.md` for a comprehensive analysis of all issues found and fixed.

## Running Tests

```bash
conda activate synth_experiments
cd /Users/carlostrujillo/Documents/GitHub/marketing-science-projects/Experimentation/site
pytest tests/test_dash_app.py -v
```

All tests should pass ✅


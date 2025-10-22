"""AI-powered summary generation for experiment results."""

from __future__ import annotations

import io
import sys
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def capture_model_summary(model: Any) -> str:
    """Capture the printed output from model.summary().
    
    Args:
        model: The causalpy model object with a summary() method
        
    Returns:
        The captured string output from summary()
    """
    try:
        # Redirect stdout to capture the print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Call summary() which prints but returns None
        model.summary()
        
        # Get the captured output
        captured_text = captured_output.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        return captured_text
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        logger.error(f"Failed to capture model summary: {e}")
        return ""


def generate_ai_summary(
    client: Any,
    results: Dict[str, Any],
    model: Any,
) -> Optional[str]:
    """Generate an AI-powered interpretation of experiment results.
    
    Args:
        client: OpenAI client instance
        results: Dictionary containing experiment results
        model: The causalpy model object
        
    Returns:
        AI-generated summary text or None if generation fails
    """
    try:
        logger.info("=== Generating AI summary ===")
        
        # Extract metrics from results
        summary_mean = results.get("summary_mean", {})
        summary_cumulative = results.get("summary_cumulative", {})
        
        mean_effect = summary_mean.get("Impact", {}).get("value", "N/A")
        mean_intervals = summary_mean.get("Impact", {}).get("subtitle", "N/A")
        prediction_y = summary_mean.get("Estimated", {}).get("value", "N/A")
        actual_value = summary_mean.get("Actual", {}).get("value", "N/A")
        
        cumulative_effect = summary_cumulative.get("Impact", {}).get("value", "N/A")
        cumulative_intervals = summary_cumulative.get("Impact", {}).get("subtitle", "N/A")
        
        # Capture the model summary output
        summary_coefficients = capture_model_summary(model)
        
        # Format the prompt
        input_experiment = f"""
The experiment got a mean effect of {mean_effect} with a 94% interval of {mean_intervals}.
The estimated value in the absence of intervention was {prediction_y}, and the actual value was {actual_value}.

The cumulative results say an effect of {cumulative_effect} with a 94% interval of {cumulative_intervals}.

The coefficients of each of the control units are:
{summary_coefficients}
"""
        
        logger.info(f"Calling OpenAI API with input length: {len(input_experiment)}")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant that creates an overview with opinion based on the results of a quasi experiment, dictating how much do you think the user should consider it truth based only on the inputs given. Use a two paragraph format without markdown style, only plain text."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_experiment
                        }
                    ]
                }
            ],
        )
        
        ai_summary = response.choices[0].message.content
        logger.info(f"AI summary generated successfully, length: {len(ai_summary)}")
        
        return ai_summary
        
    except Exception as e:
        logger.error(f"Failed to generate AI summary: {e}")
        logger.exception("AI summary generation error details:")
        return None


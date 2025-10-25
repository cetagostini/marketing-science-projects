"""Experiment chat service for interactive conversations about experiment results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ExperimentChatService:
    """Service for chatting about experiment results with context."""
    
    def __init__(self, client):
        """Initialize the experiment chat service.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
    
    def format_experiment_context(self, experiment: Dict[str, Any]) -> str:
        """Format experiment artifacts into a context string for the LLM.
        
        Args:
            experiment: Dictionary containing experiment data and results
            
        Returns:
            Formatted context string with all relevant experiment information
        """
        context_parts = []
        
        # Basic experiment information
        context_parts.append("=== EXPERIMENT OVERVIEW ===")
        context_parts.append(f"Experiment Name: {experiment.get('name', 'N/A')}")
        context_parts.append(f"User Request: {experiment.get('message', 'N/A')}")
        context_parts.append(f"Intervention Start Date: {experiment.get('start_date', 'N/A')}")
        context_parts.append(f"Intervention End Date: {experiment.get('end_date', 'N/A')}")
        context_parts.append(f"Dataset: {experiment.get('dataset_name', 'N/A')}")
        context_parts.append("")
        
        results = experiment.get("results", {})
        artifacts = results.get("artifacts", {})
        
        # Model information
        context_parts.append("=== MODEL SELECTION ===")
        selected_model = artifacts.get("selected_model", "N/A")
        context_parts.append(f"Selected Model: {selected_model}")
        
        model_reasoning = artifacts.get("model_selection_reasoning", [])
        if model_reasoning:
            context_parts.append("Model Selection Reasoning:")
            for step in model_reasoning:
                context_parts.append(f"  - {step}")
        context_parts.append("")
        
        # Control selection information
        auto_control = artifacts.get("auto_control_selection")
        if auto_control:
            context_parts.append("=== CONTROL GROUP SELECTION ===")
            context_parts.append(f"Selected Controls: {auto_control.get('selected_columns', [])}")
            reasoning = auto_control.get("reasoning", [])
            if reasoning:
                context_parts.append("Selection Reasoning:")
                for step in reasoning:
                    context_parts.append(f"  - {step}")
            context_parts.append("")
        
        # Mean effect summary
        summary_mean = results.get("summary_mean", results.get("summary", {}))
        if summary_mean:
            context_parts.append("=== MEAN EFFECT SUMMARY ===")
            context_parts.append("Average daily effect during intervention period:")
            for label, metric_data in summary_mean.items():
                if isinstance(metric_data, dict):
                    value = metric_data.get("value", "N/A")
                    subtitle = metric_data.get("subtitle", "")
                    context_parts.append(f"  {label}: {value}")
                    if subtitle:
                        context_parts.append(f"    ({subtitle})")
                else:
                    context_parts.append(f"  {label}: {metric_data}")
            context_parts.append("")
        
        # Cumulative effect summary
        summary_cumulative = results.get("summary_cumulative", {})
        if summary_cumulative:
            context_parts.append("=== CUMULATIVE EFFECT SUMMARY ===")
            context_parts.append("Total accumulated effect during intervention period:")
            for label, metric_data in summary_cumulative.items():
                if isinstance(metric_data, dict):
                    value = metric_data.get("value", "N/A")
                    subtitle = metric_data.get("subtitle", "")
                    context_parts.append(f"  {label}: {value}")
                    if subtitle:
                        context_parts.append(f"    ({subtitle})")
                else:
                    context_parts.append(f"  {label}: {metric_data}")
            context_parts.append("")
        
        # Sensitivity analysis
        sensitivity = results.get("sensitivity_analysis")
        if sensitivity:
            context_parts.append("=== SENSITIVITY ANALYSIS (PLACEBO TESTING) ===")
            if sensitivity.get("success"):
                stats = sensitivity.get("statistics", {})
                context_parts.append("Placebo tests validate model robustness by running the model on pre-intervention periods.")
                if stats:
                    context_parts.append(f"  Placebo Mean: {stats.get('mean', 'N/A')}")
                    context_parts.append(f"  Placebo Std Dev: {stats.get('std', 'N/A')}")
                    q_95 = stats.get('quantiles_95', [])
                    if q_95:
                        context_parts.append(f"  95% CI: [{q_95[0]}, {q_95[1]}]")
                    normality = stats.get('normality_tests', {})
                    if normality:
                        shapiro_p = normality.get('shapiro', {}).get('p_value', 'N/A')
                        context_parts.append(f"  Normality test p-value: {shapiro_p}")
            else:
                error_msg = sensitivity.get("error", "Unknown error")
                context_parts.append(f"Sensitivity analysis could not be completed: {error_msg}")
            context_parts.append("")
        
        # AI-generated summary
        ai_summary = results.get("ai_summary")
        if ai_summary:
            context_parts.append("=== AI-GENERATED INTERPRETATION ===")
            context_parts.append(ai_summary)
            context_parts.append("")
        
        # Intervention configuration
        context_parts.append("=== TECHNICAL DETAILS ===")
        context_parts.append(f"Target Variable: {results.get('target_var', 'N/A')}")
        control_group = results.get('control_group', [])
        if control_group:
            context_parts.append(f"Control Variables: {', '.join(control_group)}")
        extra_vars = results.get('extra_variables')
        if extra_vars:
            context_parts.append(f"Additional Variables: {', '.join(extra_vars)}")
        
        return "\n".join(context_parts)
    
    def chat_with_context(
        self,
        experiment: Dict[str, Any],
        messages: List[Dict[str, str]],
        user_question: str
    ) -> str:
        """Generate a response to user question with experiment context.
        
        Args:
            experiment: Dictionary containing experiment data and results
            messages: List of previous chat messages (for history)
            user_question: The user's current question
            
        Returns:
            Assistant's response as a string
        """
        # Format experiment context
        context = self.format_experiment_context(experiment)
        
        # Build system prompt with context
        system_prompt = (
            "You are an expert assistant helping users understand their quasi-experimental analysis results. "
            "You have access to detailed information about their experiment below. "
            "Use this context to provide accurate, insightful answers. "
            "When explaining statistical concepts, be clear but not overly technical unless asked. "
            "If asked about details not in the context, acknowledge the limitation.\n\n"
            f"{context}"
        )
        
        # Build message history
        conversation = [{"role": "system", "content": system_prompt}]
        
        # Add previous messages (conversation history)
        for msg in messages:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current question
        conversation.append({
            "role": "user",
            "content": user_question
        })
        
        try:
            logger.info(f"Sending chat request with {len(messages)} previous messages")
            
            response = self.client.responses.create(
                model="gpt-4o-mini",
                input=conversation
            )
            
            assistant_response = response.output_text
            logger.info(f"Received response: {len(assistant_response)} characters")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}"


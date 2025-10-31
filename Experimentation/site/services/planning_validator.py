"""Planning validation service for validating user questions about experiment planning."""

from __future__ import annotations

import json
import logging
from typing import Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ResponseValidator(BaseModel):
    """Pydantic model that validates the response of the user."""
    should_continue: bool
    reason: str


class PlanningValidator:
    """Validates user questions to ensure they relate to experiment planning and quasi-experimental methods."""
    
    def __init__(self, client):
        """Initialize the planning validator.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
    
    def validate_question(self, question: str) -> Tuple[bool, str]:
        """Validate if a user question is relevant to experiment planning and quasi-experimental methods.
        
        Args:
            question: The user's question to validate
            
        Returns:
            Tuple of (should_continue, reason):
                - should_continue: True if question is valid, False otherwise
                - reason: Explanation if question is invalid, empty string if valid
        """
        if not question or not question.strip():
            return False, "Question cannot be empty."
        
        try:
            logger.info(f"Validating planning question: {question[:100]}...")
            
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that validates user questions. "
                            "Determine if the question relates to experiment planning, experimental design, "
                            "quasi-experimental methods, causal inference methodology, statistical approach selection, "
                            "sample size considerations, control group selection strategies, validity concerns, "
                            "or general methodological advice for running experiments. "
                            "If the question is about these topics, set should_continue to True. "
                            "If the question is about unrelated topics (like weather, sports, general knowledge, "
                            "specific data analysis, or questions that require actual experimental data), "
                            "set should_continue to False and provide a brief, friendly reason explaining that "
                            "you can only help with questions about planning and designing quasi-experimental studies."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                text_format=ResponseValidator,
            )
            
            parsed_response = json.loads(response.output_text)
            should_continue = parsed_response.get("should_continue", False)
            reason = parsed_response.get("reason", "")
            
            logger.info(f"Validation result: should_continue={should_continue}")
            if not should_continue:
                logger.info(f"Rejection reason: {reason}")
            
            return should_continue, reason
            
        except Exception as e:
            logger.error(f"Error validating planning question: {e}")
            # On error, allow the question through (fail open)
            return True, ""


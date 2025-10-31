"""LLM-powered model selection for quasi-experimental designs."""

from __future__ import annotations

import json
import logging
from typing import Any, Tuple, List, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class Step(BaseModel):
    """A reasoning step in the model selection process."""
    
    explanation: str
    output: str


class QuasiExperimentSelector(BaseModel):
    """Pydantic response schema for model selection."""
    
    steps: List[Step]
    final_answer: str


class ModelSelectionError(RuntimeError):
    """Raised when model selection fails after all retries."""


class ModelSelector:
    """Encapsulates LLM calls for selecting the appropriate causal inference model."""
    
    VALID_MODELS = ["cp.SyntheticControl", "cp.InterruptedTimeSeries"]
    
    def __init__(
        self,
        client: Any,
        model: str = "gpt-4.1",
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._system_prompt = (
            "You are a helpful assistant that decides the best quasi-experimental model "
            "based on the user use case. You reason about the pros and cons and decide "
            "based on your best knowledge. Your final answer must be exactly one of: "
            "cp.SyntheticControl | cp.InterruptedTimeSeries"
        )
    
    def _parse_response(self, response: Any) -> QuasiExperimentSelector:
        """Parse and validate the LLM response."""
        try:
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise ModelSelectionError("LLM response was not valid JSON") from exc
        
        try:
            return QuasiExperimentSelector.model_validate(payload)
        except ValidationError as exc:
            raise ModelSelectionError("LLM response schema validation failed") from exc
    
    def _is_valid_model(self, model_string: str) -> bool:
        """Check if the model string is one of the valid options."""
        return model_string in self.VALID_MODELS
    
    def select_model(self, user_message: str) -> Tuple[str, List[dict]]:
        """Select the appropriate model based on user's experiment description.
        
        Args:
            user_message: The user's description of their experiment
            
        Returns:
            Tuple of (model_string, reasoning_steps)
            where model_string is like "cp.SyntheticControl"
            and reasoning_steps is a list of dicts with explanation and output
            
        Raises:
            ModelSelectionError: If model selection fails after all retries
        """
        logger.info("=== select_model called ===")
        logger.info(f"User message length: {len(user_message)}")
        logger.info(f"Using model: {self._model}")
        
        for attempt in range(self._max_retries):
            logger.info(f"Model selection attempt {attempt + 1}/{self._max_retries}")
            
            try:
                response = self._client.responses.parse(
                    model=self._model,
                    input=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    text_format=QuasiExperimentSelector,
                )
                
                logger.info("LLM response received")
                parsed = self._parse_response(response)
                
                logger.info(f"Parsed response with final_answer: {parsed.final_answer}")
                logger.info(f"Number of reasoning steps: {len(parsed.steps)}")
                
                if self._is_valid_model(parsed.final_answer):
                    logger.info(f"Valid model selected: {parsed.final_answer}")
                    
                    # Convert steps to dict format for storage
                    reasoning_steps = [
                        {"explanation": step.explanation, "output": step.output}
                        for step in parsed.steps
                    ]
                    
                    return parsed.final_answer, reasoning_steps
                else:
                    logger.warning(
                        f"Invalid model '{parsed.final_answer}' returned. "
                        f"Valid options: {self.VALID_MODELS}"
                    )
                    
            except Exception as e:
                logger.error(f"Error in select_model attempt {attempt + 1}: {e}")
                if attempt == self._max_retries - 1:
                    # Last attempt failed
                    raise ModelSelectionError(
                        f"Model selection failed after {self._max_retries} attempts: {e}"
                    ) from e
        
        # Should not reach here, but for type safety
        raise ModelSelectionError(
            f"Model selection failed after {self._max_retries} attempts"
        )


def select_model(client: Any, user_message: str) -> Tuple[str, List[dict]]:
    """Convenience function for model selection.
    
    Args:
        client: OpenAI client instance
        user_message: The user's description of their experiment
        
    Returns:
        Tuple of (model_string, reasoning_steps)
    """
    selector = ModelSelector(client)
    return selector.select_model(user_message)


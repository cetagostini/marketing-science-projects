"""LLM-assisted extraction workflows for experiment setup."""

from __future__ import annotations

import json
import logging
from typing import Any
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class DataExtractor(BaseModel):
    """Pydantic response schema for experiment parameters."""

    intervention_start_date: str
    intervention_end_date: str
    target_var: str
    control_group: list[str]


class DateColumnExtractor(BaseModel):
    """Pydantic response schema for identifying the date column."""

    date_column: str


@dataclass
class LLMExtractionConfig:
    """Configuration for LLM extraction services."""

    data_model: str = "gpt-4o"  # Fixed: was "gpt-4.1" which is not a valid model
    system_prompt: str = (
        "You are a helpful assistant that extracts data from text on the format required."
    )
    date_system_prompt: str = (
        "You are a helpful assistant that extracts any dates column name on the format required based on the dataset information."
    )


class ExtractionError(RuntimeError):
    """Raised when extraction via LLM fails."""


class LLMExtractor:
    """Encapsulates LLM calls for extracting structured experiment data."""

    def __init__(self, client: Any, config: Optional[LLMExtractionConfig] = None) -> None:
        self._client = client
        self._config = config or LLMExtractionConfig()

    def _parse_response(self, response: Any, model_cls: type[BaseModel]) -> BaseModel:
        try:
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ExtractionError("LLM response was not valid JSON") from exc

        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError("LLM response schema validation failed") from exc

    def extract_experiment_details(self, question: str) -> DataExtractor:
        logger.info("=== extract_experiment_details called ===")
        logger.info(f"Question: {question[:200]}...")
        logger.info(f"Using model: {self._config.data_model}")
        
        try:
            response = self._client.responses.parse(
                model=self._config.data_model,
                input=[
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": question},
                ],
                text_format=DataExtractor,
            )
            logger.info("LLM response received")
            result = self._parse_response(response, DataExtractor)
            logger.info(f"Extracted: start={result.intervention_start_date}, end={result.intervention_end_date}, target={result.target_var}, controls={result.control_group}")
            return result
        except Exception as e:
            logger.error(f"Error in extract_experiment_details: {e}")
            raise

    def extract_date_column(self, df: pd.DataFrame) -> str:
        logger.info("=== extract_date_column called ===")
        columns_as_list = df.columns.tolist()
        logger.info(f"Available columns: {columns_as_list}")

        question = (
            "Read the following dataset and determinate which one is the date column.\n\n"
            f"{columns_as_list}"
        )

        try:
            response = self._client.responses.parse(
                model=self._config.data_model,
                input=[
                    {"role": "system", "content": self._config.date_system_prompt},
                    {"role": "user", "content": question},
                ],
                text_format=DateColumnExtractor,
            )

            parsed = self._parse_response(response, DateColumnExtractor)
            logger.info(f"LLM suggested date column: {parsed.date_column}")
            
            if parsed.date_column not in df.columns:
                logger.error(f"Date column '{parsed.date_column}' not in dataframe columns")
                raise ExtractionError(
                    f"LLM suggested '{parsed.date_column}' which is not present in dataframe columns"
                )
            return parsed.date_column
        except Exception as e:
            logger.error(f"Error in extract_date_column: {e}")
            raise


def normalise_control_codes(values: Iterable[str]) -> list[str]:
    """Normalise control group identifiers, trimming whitespace and removing blanks."""

    normalised: list[str] = []
    for value in values:
        if value is None:
            continue
        normalised_value = str(value).strip()
        if normalised_value:
            normalised.append(normalised_value)
    return normalised



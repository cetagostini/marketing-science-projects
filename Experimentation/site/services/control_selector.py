"""Automatic control group selection using Bayesian variable selection."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import MaxAbsScaler

logger = logging.getLogger(__name__)


class DataIdentifier(BaseModel):
    """Schema for data structure identification.
    
    Attributes
    ----------
    date_column : str
        The column name of the date column.
    numerical_columns : list[str]
        The list of numerical columns.
    categorical_columns : list[str] | None
        The list of categorical columns (optional).
    """
    date_column: str
    numerical_columns: list[str]
    categorical_columns: list[str] | None = None


class ControlSelectionError(RuntimeError):
    """Raised when automatic control selection fails."""


class FeatureSelector:
    """
    A class for selecting features using Bayesian variable selection with PyMC.
    """
    
    def __init__(self):
        self.x_scaler = MaxAbsScaler()
        self.y_scaler = MaxAbsScaler()
        self.model = None
        self.idata = None
        self.selected_columns = []
    
    def _scale_data(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Scale the training data using MaxAbsScaler."""
        x_train_scaled = self.x_scaler.fit_transform(x_train.values)
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        return x_train_scaled, y_train_scaled
    
    def _build_model(self, X: np.ndarray, y: np.ndarray, coords: dict) -> pm.Model:
        """Build the Bayesian model for feature selection."""
        with pm.Model(coords=coords) as model:
            # Define priors for the intercept and coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            betas = pm.Laplace('betas', mu=0, b=1, dims="predictors")

            # Linear model
            mu = intercept + pm.math.dot(X, betas)

            # Likelihood
            sigma = pm.HalfNormal('sigma', sigma=1)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y, dims="dates")

        return model
    
    def _sample_model(self):
        """Sample from the model using MCMC."""
        with self.model:
            self.idata = pm.sample(
                draws=200,
                chains=4,
                tune=800,
                target_accept=0.84,
                progressbar=False,  # Disable progress bar for cleaner logs
            )
    
    def _select_features(self, predictors_names: list[str], hdi_prob: float = 0.84) -> tuple[list[str], Dict[str, Dict[str, float]]]:
        """Select features based on credible intervals.
        
        Returns
        -------
        tuple
            (selected_columns, beta_statistics)
        """
        # Extracting the beta coefficients
        betas_trace = self.idata.posterior['betas']

        # Mean of the betas across samples
        betas_mean = betas_trace.mean(dim=['chain', 'draw']).values

        # Determine which coefficients are significantly non-zero
        credible_intervals = az.hdi(self.idata, hdi_prob=hdi_prob)

        # Extract the 'betas' variable from the credible intervals dataset
        betas_credible_intervals = credible_intervals['betas']

        selected_columns = []
        beta_statistics = {}
        
        for idx, column in enumerate(predictors_names):
            lower, upper = betas_credible_intervals.sel(predictors=column).values
            mean_val = float(betas_mean[idx])
            
            # Store statistics for all columns
            beta_statistics[column] = {
                "mean": mean_val,
                "lower_hdi": float(lower),
                "upper_hdi": float(upper),
            }
            
            # Select if credible interval doesn't include zero
            if not (lower <= 0 <= upper):
                selected_columns.append(column)

        return selected_columns, beta_statistics
    
    def select_features(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        date_column_values: np.ndarray,
    ) -> tuple[list[str], Dict[str, Dict[str, float]]]:
        """
        Main method to select features using Bayesian variable selection.
        
        Parameters
        ----------
        x_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target variable
        date_column_values : array-like
            Date column values for coordinates
            
        Returns
        -------
        tuple
            (selected_columns, beta_statistics)
        """
        logger.info(f"Starting feature selection with {len(x_train.columns)} potential controls")
        
        # Scale the data
        x_train_scaled, y_train_scaled = self._scale_data(x_train, y_train)
        
        # Get names for predictors
        predictors_names = x_train.columns.tolist()
        coordinates = {"predictors": predictors_names, "dates": date_column_values}
        
        # Build the model
        self.model = self._build_model(X=x_train_scaled, y=y_train_scaled, coords=coordinates)
        
        # Sample from the model
        logger.info("Running PyMC sampling (200 draws, 800 tune, 4 chains)...")
        self._sample_model()
        logger.info("PyMC sampling completed")
        
        # Select features
        self.selected_columns, beta_stats = self._select_features(predictors_names)
        
        logger.info(f"Selected {len(self.selected_columns)} control columns: {self.selected_columns}")
        return self.selected_columns, beta_stats


class ControlSelector:
    """Main orchestrator for automatic control group selection.
    
    Uses LLM to understand data structure and PyMC for Bayesian feature selection.
    """
    
    def __init__(self, client: Any, model: str = "gpt-4o"):
        """
        Initialize the control selector.
        
        Parameters
        ----------
        client : Any
            OpenAI client for LLM calls
        model : str
            LLM model to use (default: gpt-4o)
        """
        self._client = client
        self._model = model
    
    def _describe_data(self, df: pd.DataFrame) -> str:
        """Uses LLM to analyze CSV data and provide a summary.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataset to analyze
            
        Returns
        -------
        str
            LLM-generated description of the data
        """
        logger.info("Generating data description using LLM...")
        csv_content = df.to_string()
        
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant that analyzes CSV data and provides a summary of the columns, data points, and values. Here is the data: \n\n{csv_content}"
                    },
                    {
                        "role": "user",
                        "content": "Analyze this CSV data and provide a summary of the content."
                    }
                ]
            )
            description = response.output_text
            logger.info(f"Data description generated ({len(description)} chars)")
            return description
        except Exception as e:
            logger.error(f"Failed to generate data description: {e}")
            raise ControlSelectionError(f"Failed to describe data: {str(e)}") from e
    
    def _identify_data_structure(self, description: str) -> DataIdentifier:
        """Uses LLM to parse data structure from description.
        
        Parameters
        ----------
        description : str
            LLM-generated description of the data
            
        Returns
        -------
        DataIdentifier
            Parsed data structure information
        """
        logger.info("Identifying data structure using LLM...")
        
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that based on the description of the data from the previous assistant, identify the date column, the categorical columns and the numerical columns."
                    },
                    {
                        "role": "user",
                        "content": description
                    }
                ],
                text_format=DataIdentifier,
            )
            
            payload = json.loads(response.output_text)
            data_identifier = DataIdentifier.model_validate(payload)
            
            logger.info(f"Identified date column: {data_identifier.date_column}")
            logger.info(f"Identified {len(data_identifier.numerical_columns)} numerical columns")
            
            return data_identifier
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse data structure: {e}")
            raise ControlSelectionError(f"Failed to identify data structure: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in data structure identification: {e}")
            raise ControlSelectionError(f"Data structure identification failed: {str(e)}") from e
    
    def _select_features(
        self,
        df: pd.DataFrame,
        target_var: str,
        intervention_start_date: str,
        date_column: str,
        numerical_columns: list[str],
    ) -> tuple[list[str], Dict[str, Dict[str, float]]]:
        """Runs PyMC feature selection on pre-intervention data.
        
        Parameters
        ----------
        df : pd.DataFrame
            The full dataset
        target_var : str
            The target variable column name
        intervention_start_date : str
            The intervention start date (ISO format)
        date_column : str
            The date column name
        numerical_columns : list[str]
            List of numerical columns to consider
            
        Returns
        -------
        tuple
            (selected_columns, beta_statistics)
        """
        logger.info("Preparing data for feature selection...")
        
        # Filter pre-intervention data
        try:
            pre_intervention_df = df.query(f"{date_column} < '{intervention_start_date}'")
            
            if pre_intervention_df.empty:
                raise ControlSelectionError(
                    f"No pre-intervention data available before {intervention_start_date}"
                )
            
            logger.info(f"Pre-intervention data: {len(pre_intervention_df)} rows")
            
            # Check if target variable is in numerical columns
            if target_var not in numerical_columns:
                raise ControlSelectionError(
                    f"Target variable '{target_var}' not found in numerical columns"
                )
            
            # Prepare features (exclude target variable)
            available_controls = [col for col in numerical_columns if col != target_var]
            
            if not available_controls:
                raise ControlSelectionError(
                    f"No numerical columns available as controls (target: {target_var})"
                )
            
            x_train = pre_intervention_df[available_controls]
            y_train = pre_intervention_df[target_var]
            date_values = pre_intervention_df[date_column].values
            
            logger.info(f"Training with {len(available_controls)} potential control variables")
            
            # Run feature selection
            selector = FeatureSelector()
            selected_columns, beta_stats = selector.select_features(
                x_train=x_train,
                y_train=y_train,
                date_column_values=date_values,
            )
            
            if not selected_columns:
                logger.warning("No features selected by Bayesian model, using all available controls")
                # If no features pass the credible interval test, use all available
                selected_columns = available_controls
            
            return selected_columns, beta_stats
            
        except Exception as e:
            if isinstance(e, ControlSelectionError):
                raise
            logger.error(f"Feature selection failed: {e}")
            raise ControlSelectionError(f"PyMC feature selection failed: {str(e)}") from e
    
    def select_controls(
        self,
        df: pd.DataFrame,
        target_var: str,
        intervention_start_date: str,
    ) -> Dict[str, Any]:
        """Main entry point for automatic control selection.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataset
        target_var : str
            The target variable column name
        intervention_start_date : str
            The intervention start date (ISO format)
            
        Returns
        -------
        dict
            {
                "selected_columns": list[str],
                "beta_statistics": dict,
                "data_description": str,
                "data_structure": dict,
            }
        """
        logger.info("=== Starting automatic control selection ===")
        logger.info(f"Target variable: {target_var}")
        logger.info(f"Intervention start: {intervention_start_date}")
        
        try:
            # Step 1: Describe the data
            description = self._describe_data(df)
            
            # Step 2: Identify data structure
            data_structure = self._identify_data_structure(description)
            
            # Step 3: Select features using PyMC
            selected_columns, beta_stats = self._select_features(
                df=df,
                target_var=target_var,
                intervention_start_date=intervention_start_date,
                date_column=data_structure.date_column,
                numerical_columns=data_structure.numerical_columns,
            )
            
            result = {
                "selected_columns": selected_columns,
                "beta_statistics": beta_stats,
                "data_description": description,
                "data_structure": data_structure.model_dump(),
            }
            
            logger.info("=== Automatic control selection completed successfully ===")
            return result
            
        except ControlSelectionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in control selection: {e}")
            raise ControlSelectionError(f"Control selection failed: {str(e)}") from e


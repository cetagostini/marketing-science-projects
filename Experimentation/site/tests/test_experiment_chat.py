"""Tests for experiment chat service."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from services.experiment_chat import ExperimentChatService


class TestExperimentChatService:
    """Tests for ExperimentChatService."""
    
    def test_format_experiment_context_basic_info(self):
        """Test that basic experiment info is included in context."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment 1",
            "message": "Test intervention effect",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "dataset_name": "test_data.csv",
            "results": {}
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "Test Experiment 1" in context
        assert "Test intervention effect" in context
        assert "2024-01-01" in context
        assert "2024-01-31" in context
        assert "test_data.csv" in context
    
    def test_format_experiment_context_with_model_info(self):
        """Test that model selection info is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "artifacts": {
                    "selected_model": "cp.SyntheticControl",
                    "model_selection_reasoning": [
                        "Model fits data well",
                        "Pre-intervention parallel trends"
                    ]
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "cp.SyntheticControl" in context
        assert "Model fits data well" in context
        assert "Pre-intervention parallel trends" in context
    
    def test_format_experiment_context_with_summary_mean(self):
        """Test that mean effect summary is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "summary_mean": {
                    "Effect": {"value": "12.5", "subtitle": "units"},
                    "P-Value": {"value": "0.001", "subtitle": "highly significant"}
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "MEAN EFFECT SUMMARY" in context
        assert "Effect" in context
        assert "12.5" in context
        assert "P-Value" in context
        assert "0.001" in context
    
    def test_format_experiment_context_with_cumulative_summary(self):
        """Test that cumulative effect summary is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "summary_cumulative": {
                    "Total Effect": {"value": "375", "subtitle": "cumulative units"}
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "CUMULATIVE EFFECT SUMMARY" in context
        assert "Total Effect" in context
        assert "375" in context
    
    def test_format_experiment_context_with_sensitivity_analysis(self):
        """Test that sensitivity analysis is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "sensitivity_analysis": {
                    "success": True,
                    "statistics": {
                        "mean": 0.05,
                        "std": 2.3,
                        "quantiles_95": [-4.5, 4.7],
                        "normality_tests": {
                            "shapiro": {"p_value": 0.82}
                        }
                    }
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "SENSITIVITY ANALYSIS" in context
        assert "0.05" in context
        assert "2.3" in context
        assert "0.82" in context
    
    def test_format_experiment_context_with_failed_sensitivity(self):
        """Test formatting when sensitivity analysis failed."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "sensitivity_analysis": {
                    "success": False,
                    "error": "Not enough pre-intervention data"
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "SENSITIVITY ANALYSIS" in context
        assert "Not enough pre-intervention data" in context
    
    def test_format_experiment_context_with_ai_summary(self):
        """Test that AI summary is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "ai_summary": "The intervention had a significant positive effect..."
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "AI-GENERATED INTERPRETATION" in context
        assert "significant positive effect" in context
    
    def test_format_experiment_context_with_control_selection(self):
        """Test that auto control selection info is included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "artifacts": {
                    "auto_control_selection": {
                        "selected_columns": ["control_1", "control_2"],
                        "reasoning": [
                            "High correlation with target",
                            "Good pre-trend alignment"
                        ]
                    }
                }
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "CONTROL GROUP SELECTION" in context
        assert "control_1" in context
        assert "control_2" in context
        assert "High correlation with target" in context
    
    def test_format_experiment_context_with_technical_details(self):
        """Test that technical details are included."""
        mock_client = MagicMock()
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "results": {
                "target_var": "sales",
                "control_group": ["region_a", "region_b"],
                "extra_variables": ["seasonality", "trend"]
            }
        }
        
        context = service.format_experiment_context(experiment)
        
        assert "TECHNICAL DETAILS" in context
        assert "sales" in context
        assert "region_a" in context
        assert "region_b" in context
        assert "seasonality" in context
        assert "trend" in context
    
    def test_chat_with_context_simple_question(self):
        """Test chat response generation with simple question."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = "The intervention had a positive effect of 12.5 units."
        mock_client.responses.create.return_value = mock_response
        
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "Test Experiment",
            "message": "Test the effect",
            "results": {
                "summary_mean": {
                    "Effect": {"value": "12.5"}
                }
            }
        }
        
        response = service.chat_with_context(
            experiment,
            [],
            "What was the effect?"
        )
        
        assert response == "The intervention had a positive effect of 12.5 units."
        mock_client.responses.create.assert_called_once()
    
    def test_chat_with_context_with_history(self):
        """Test chat with conversation history."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = "Yes, it was statistically significant with p < 0.001."
        mock_client.responses.create.return_value = mock_response
        
        service = ExperimentChatService(mock_client)
        
        experiment = {"name": "Test Experiment", "results": {}}
        
        messages = [
            {"role": "user", "content": "What was the effect?"},
            {"role": "assistant", "content": "The effect was 12.5 units."}
        ]
        
        response = service.chat_with_context(
            experiment,
            messages,
            "Was it significant?"
        )
        
        # Check that history was included
        call_args = mock_client.responses.create.call_args
        conversation = call_args[1]["input"]
        
        # Should have system prompt + 2 history messages + new question = 4 total
        assert len(conversation) == 4
        assert conversation[0]["role"] == "system"
        assert conversation[1]["role"] == "user"
        assert conversation[2]["role"] == "assistant"
        assert conversation[3]["role"] == "user"
    
    def test_chat_with_context_uses_correct_model(self):
        """Test that chat uses the correct model."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_client.responses.create.return_value = mock_response
        
        service = ExperimentChatService(mock_client)
        
        experiment = {"name": "Test", "results": {}}
        service.chat_with_context(experiment, [], "Test question?")
        
        call_args = mock_client.responses.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
    
    def test_chat_with_context_includes_system_prompt(self):
        """Test that system prompt includes experiment context."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_client.responses.create.return_value = mock_response
        
        service = ExperimentChatService(mock_client)
        
        experiment = {
            "name": "My Experiment",
            "results": {"target_var": "sales"}
        }
        
        service.chat_with_context(experiment, [], "Test?")
        
        call_args = mock_client.responses.create.call_args
        conversation = call_args[1]["input"]
        system_prompt = conversation[0]["content"]
        
        assert "expert assistant" in system_prompt.lower()
        assert "My Experiment" in system_prompt
        assert "sales" in system_prompt
    
    def test_chat_with_context_api_error(self):
        """Test error handling when API call fails."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("API Error")
        
        service = ExperimentChatService(mock_client)
        
        experiment = {"name": "Test", "results": {}}
        response = service.chat_with_context(experiment, [], "Test?")
        
        assert "error" in response.lower()
        assert "API Error" in response


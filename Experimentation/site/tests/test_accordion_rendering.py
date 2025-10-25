"""Tests for Mantine accordion rendering in experiment reports."""

import pytest
from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# Import the functions to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dash_app import _build_posterior_accordion, _build_sensitivity_accordion, _render_experiment_detail


class TestPosteriorAccordion:
    """Tests for the posterior effect distributions accordion."""

    def test_accordion_renders_with_mean_effect(self):
        """Test that accordion renders when mean effect data is present."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        posterior_cumulative = []
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        assert accordion is not None
        assert isinstance(accordion, dmc.Accordion)
        assert accordion.disableChevronRotation is True
        assert accordion.variant == "separated"

    def test_accordion_renders_with_cumulative_effect(self):
        """Test that accordion renders when cumulative effect data is present."""
        posterior_mean = []
        posterior_cumulative = [10.5, 12.0, 14.5, 16.0]
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        assert accordion is not None
        assert isinstance(accordion, dmc.Accordion)

    def test_accordion_renders_with_both_effects(self):
        """Test that accordion renders when both mean and cumulative data are present."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        posterior_cumulative = [10.5, 12.0, 14.5, 16.0]
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        assert accordion is not None
        assert isinstance(accordion, dmc.Accordion)

    def test_accordion_returns_none_when_no_data(self):
        """Test that accordion returns None when no data is available."""
        posterior_mean = []
        posterior_cumulative = []
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        assert accordion is None

    def test_accordion_has_correct_icon(self):
        """Test that accordion uses the bar-chart icon."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        posterior_cumulative = []
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        # Check that accordion has children
        assert len(accordion.children) > 0
        accordion_item = accordion.children[0]
        
        # Check that accordion item has children (control and panel)
        assert len(accordion_item.children) >= 1
        accordion_control = accordion_item.children[0]
        
        # Check that control has an icon
        assert accordion_control.icon is not None
        assert isinstance(accordion_control.icon, DashIconify)
        assert accordion_control.icon.icon == "fluent-emoji-flat:bar-chart"

    def test_accordion_has_correct_title(self):
        """Test that accordion has the correct title."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        posterior_cumulative = []
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        accordion_item = accordion.children[0]
        accordion_control = accordion_item.children[0]
        
        # Check the title
        assert accordion_control.children == "Posterior Effect Distributions"

    def test_accordion_panel_contains_graphs(self):
        """Test that accordion panel contains the expected graphs."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        posterior_cumulative = [10.5, 12.0, 14.5, 16.0]
        
        accordion = _build_posterior_accordion(posterior_mean, posterior_cumulative)
        
        accordion_item = accordion.children[0]
        accordion_panel = accordion_item.children[1]
        panel_content = accordion_panel.children
        
        # Count the number of graphs in the panel
        graph_count = sum(1 for item in panel_content if isinstance(item, dcc.Graph))
        assert graph_count == 2  # Should have both mean and cumulative graphs


class TestSensitivityAccordion:
    """Tests for the sensitivity analysis accordion."""

    def test_accordion_renders_with_successful_sensitivity(self):
        """Test that accordion renders when sensitivity analysis succeeds."""
        sensitivity_data = {
            "success": True,
            "posterior_mean": [0.1, 0.2, 0.15],
            "posterior_cumulative": [1.0, 2.0, 1.5],
            "statistics": {
                "mean": 0.15,
                "std": 0.05,
                "quantiles_95": [0.05, 0.25],
                "normality_tests": {
                    "shapiro": {"p_value": 0.8}
                }
            }
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        assert accordion is not None
        assert isinstance(accordion, dmc.Accordion)

    def test_accordion_renders_with_failed_sensitivity(self):
        """Test that accordion renders error message when sensitivity analysis fails."""
        sensitivity_data = {
            "success": False,
            "error": "Insufficient pre-intervention data"
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        assert accordion is not None
        assert isinstance(accordion, dmc.Accordion)

    def test_accordion_returns_none_when_no_sensitivity_data(self):
        """Test that accordion returns None when no sensitivity data is available."""
        accordion = _build_sensitivity_accordion(None)
        
        assert accordion is None

    def test_accordion_has_correct_icon(self):
        """Test that accordion uses the shield icon."""
        sensitivity_data = {
            "success": True,
            "posterior_mean": [0.1, 0.2, 0.15],
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        accordion_item = accordion.children[0]
        accordion_control = accordion_item.children[0]
        
        assert accordion_control.icon is not None
        assert isinstance(accordion_control.icon, DashIconify)
        assert accordion_control.icon.icon == "fluent-emoji-flat:shield"

    def test_accordion_has_correct_title(self):
        """Test that accordion has the correct title."""
        sensitivity_data = {
            "success": True,
            "posterior_mean": [0.1, 0.2, 0.15],
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        accordion_item = accordion.children[0]
        accordion_control = accordion_item.children[0]
        
        assert accordion_control.children == "Sensitivity Analysis (Placebo Testing)"

    def test_accordion_contains_statistics_cards(self):
        """Test that accordion contains metric cards with statistics."""
        sensitivity_data = {
            "success": True,
            "posterior_mean": [0.1, 0.2, 0.15],
            "statistics": {
                "mean": 0.15,
                "std": 0.05,
                "quantiles_95": [0.05, 0.25],
                "normality_tests": {
                    "shapiro": {"p_value": 0.8}
                }
            }
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        accordion_item = accordion.children[0]
        accordion_panel = accordion_item.children[1]
        panel_content = accordion_panel.children
        
        # Look for metric-grid divs
        metric_grids = [item for item in panel_content if isinstance(item, html.Div) and item.className == "metric-grid"]
        assert len(metric_grids) > 0

    def test_accordion_shows_error_message_on_failure(self):
        """Test that accordion displays error message when sensitivity fails."""
        error_msg = "Insufficient pre-intervention data"
        sensitivity_data = {
            "success": False,
            "error": error_msg
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        
        accordion_item = accordion.children[0]
        accordion_panel = accordion_item.children[1]
        panel_content = accordion_panel.children
        
        # Check that error notification exists
        error_divs = [item for item in panel_content if isinstance(item, html.Div) and item.className == "error-notification"]
        assert len(error_divs) > 0


class TestExperimentDetailIntegration:
    """Integration tests for full experiment detail rendering with accordions."""

    def test_experiment_detail_includes_accordions(self):
        """Test that _render_experiment_detail includes accordions when data is available."""
        experiment = {
            "name": "Test Experiment",
            "message": "Test experiment message",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "dataset_name": "test_data.csv",
            "results": {
                "summary": {
                    "Effect": "5.2",
                    "Relative Effect": "12.5%",
                },
                "summary_mean": {
                    "Effect": "5.2",
                    "Relative Effect": "12.5%",
                },
                "series": {"observed": [1, 2, 3], "predicted": [1.1, 2.1, 3.1]},
                "lift_table": [
                    {"Metric": "Effect", "Value": "5.2"},
                ],
                "posterior_mean_effect": [1.0, 1.5, 2.0],
                "posterior_cumulative_effect": [10.0, 15.0, 20.0],
                "sensitivity_analysis": {
                    "success": True,
                    "posterior_mean": [0.1, 0.2, 0.15],
                    "statistics": {
                        "mean": 0.15,
                        "std": 0.05,
                        "quantiles_95": [0.05, 0.25],
                        "normality_tests": {
                            "shapiro": {"p_value": 0.8}
                        }
                    }
                },
                "artifacts": {
                    "selected_model": "cp.SyntheticControl"
                }
            }
        }
        
        components = _render_experiment_detail(experiment)
        
        # Count accordions in components
        accordion_count = sum(1 for comp in components if isinstance(comp, dmc.Accordion))
        assert accordion_count == 2  # Should have 2 accordions

    def test_experiment_detail_without_posterior_data(self):
        """Test that experiment detail works without posterior data."""
        experiment = {
            "name": "Test Experiment",
            "message": "Test experiment message",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "dataset_name": "test_data.csv",
            "results": {
                "summary": {
                    "Effect": "5.2",
                },
                "summary_mean": {
                    "Effect": "5.2",
                },
                "series": {"observed": [1, 2, 3], "predicted": [1.1, 2.1, 3.1]},
                "lift_table": [
                    {"Metric": "Effect", "Value": "5.2"},
                ],
                "artifacts": {
                    "selected_model": "cp.SyntheticControl"
                }
            }
        }
        
        components = _render_experiment_detail(experiment)
        
        # Should not have any accordions
        accordion_count = sum(1 for comp in components if isinstance(comp, dmc.Accordion))
        assert accordion_count == 0

    def test_experiment_detail_with_only_sensitivity(self):
        """Test that experiment detail renders only sensitivity accordion when no posterior data."""
        experiment = {
            "name": "Test Experiment",
            "message": "Test experiment message",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "dataset_name": "test_data.csv",
            "results": {
                "summary": {
                    "Effect": "5.2",
                },
                "summary_mean": {
                    "Effect": "5.2",
                },
                "series": {"observed": [1, 2, 3], "predicted": [1.1, 2.1, 3.1]},
                "lift_table": [
                    {"Metric": "Effect", "Value": "5.2"},
                ],
                "sensitivity_analysis": {
                    "success": True,
                    "posterior_mean": [0.1, 0.2, 0.15],
                },
                "artifacts": {
                    "selected_model": "cp.SyntheticControl"
                }
            }
        }
        
        components = _render_experiment_detail(experiment)
        
        # Should have only 1 accordion (sensitivity)
        accordion_count = sum(1 for comp in components if isinstance(comp, dmc.Accordion))
        assert accordion_count == 1

    def test_ai_summary_remains_visible(self):
        """Test that AI summary is rendered outside accordions."""
        experiment = {
            "name": "Test Experiment",
            "message": "Test experiment message",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "dataset_name": "test_data.csv",
            "results": {
                "summary": {
                    "Effect": "5.2",
                },
                "summary_mean": {
                    "Effect": "5.2",
                },
                "series": {"observed": [1, 2, 3], "predicted": [1.1, 2.1, 3.1]},
                "lift_table": [
                    {"Metric": "Effect", "Value": "5.2"},
                ],
                "ai_summary": "This is a test AI summary.",
                "artifacts": {
                    "selected_model": "cp.SyntheticControl"
                }
            }
        }
        
        components = _render_experiment_detail(experiment)
        
        # Check that AI summary card is present
        ai_summary_cards = [
            comp for comp in components 
            if isinstance(comp, html.Div) and comp.className == "ai-summary-card"
        ]
        assert len(ai_summary_cards) == 1


class TestAccordionStructure:
    """Tests for accordion component structure."""

    def test_accordion_item_has_value(self):
        """Test that AccordionItem has the correct value attribute."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        
        accordion = _build_posterior_accordion(posterior_mean, [])
        accordion_item = accordion.children[0]
        
        assert accordion_item.value == "posterior"

    def test_sensitivity_accordion_item_has_value(self):
        """Test that sensitivity AccordionItem has the correct value attribute."""
        sensitivity_data = {
            "success": True,
            "posterior_mean": [0.1, 0.2, 0.15],
        }
        
        accordion = _build_sensitivity_accordion(sensitivity_data)
        accordion_item = accordion.children[0]
        
        assert accordion_item.value == "sensitivity"

    def test_accordion_has_chevron_position_left(self):
        """Test that accordion has chevron on the left."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        
        accordion = _build_posterior_accordion(posterior_mean, [])
        
        assert accordion.chevronPosition == "left"

    def test_accordion_variant_is_separated(self):
        """Test that accordion uses separated variant."""
        posterior_mean = [1.2, 1.5, 1.8, 2.0]
        
        accordion = _build_posterior_accordion(posterior_mean, [])
        
        assert accordion.variant == "separated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


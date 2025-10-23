"""Test delete button click isolation and event handling."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dash_app import show_delete_confirmation


class TestClickIsolation:
    """Test that delete button clicks are isolated from experiment card clicks."""
    
    @patch('dash_app.callback_context')
    def test_experiment_card_click_not_trigger_delete(self, mock_ctx):
        """Test that clicking experiment card does NOT trigger delete confirmation."""
        from dash.exceptions import PreventUpdate
        
        # Simulate experiment card click (wrong button type)
        mock_ctx.triggered_id = {"type": "experiment-card", "value": "Experiment 1"}
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_menu_button_click_triggers_delete(self, mock_ctx):
        """Test that clicking menu button DOES trigger delete confirmation."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        result = show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        assert result[0] == "Experiment 1"
        assert result[1] is True
    
    @patch('dash_app.callback_context')
    def test_menu_button_on_specific_experiment(self, mock_ctx):
        """Test clicking menu button on one experiment doesn't affect others."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 2"}
        
        # Multiple experiments with different click counts
        result = show_delete_confirmation(
            [0, 1, 0],  # Only Experiment 2 was clicked
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
                {"type": "experiment-menu-button", "experiment": "Experiment 3"},
            ]
        )
        
        # Should return Experiment 2
        assert result[0] == "Experiment 2"
        assert result[1] is True
    
    @patch('dash_app.callback_context')
    def test_zero_clicks_initial_state(self, mock_ctx):
        """Test that initial state with zero clicks prevents update."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # All clicks are 0 (initial page load)
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([0, 0, 0], [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
                {"type": "experiment-menu-button", "experiment": "Experiment 3"},
            ])
    
    @patch('dash_app.callback_context')
    def test_none_clicks_prevents_update(self, mock_ctx):
        """Test that None clicks prevents update."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # Clicks are None
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([None], [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
            ])
    
    @patch('dash_app.callback_context')
    def test_rapid_clicks_single_modal(self, mock_ctx):
        """Test that rapid clicks on menu button still show one modal."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # Rapid clicks would show as higher count
        result = show_delete_confirmation([3], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        # Should still return single modal trigger
        assert result[0] == "Experiment 1"
        assert result[1] is True


class TestButtonTypeValidation:
    """Test button type validation in callbacks."""
    
    @patch('dash_app.callback_context')
    def test_invalid_button_type_string(self, mock_ctx):
        """Test that invalid button type (string) raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "wrong-button-type", "experiment": "Experiment 1"}
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_experiment_card_type_prevents_delete(self, mock_ctx):
        """Test that experiment-card type prevents delete confirmation."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-card", "value": "Experiment 1"}
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_correct_button_type_allows_delete(self, mock_ctx):
        """Test that correct button type allows delete confirmation."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        result = show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        assert result[0] == "Experiment 1"
        assert result[1] is True


class TestExperimentSelection:
    """Test that correct experiment is selected for deletion."""
    
    @patch('dash_app.callback_context')
    def test_first_experiment_selection(self, mock_ctx):
        """Test selecting first experiment for deletion."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        result = show_delete_confirmation(
            [1, 0, 0],
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
                {"type": "experiment-menu-button", "experiment": "Experiment 3"},
            ]
        )
        
        assert result[0] == "Experiment 1"
    
    @patch('dash_app.callback_context')
    def test_middle_experiment_selection(self, mock_ctx):
        """Test selecting middle experiment for deletion."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 2"}
        
        result = show_delete_confirmation(
            [0, 1, 0],
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
                {"type": "experiment-menu-button", "experiment": "Experiment 3"},
            ]
        )
        
        assert result[0] == "Experiment 2"
    
    @patch('dash_app.callback_context')
    def test_last_experiment_selection(self, mock_ctx):
        """Test selecting last experiment for deletion."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 3"}
        
        result = show_delete_confirmation(
            [0, 0, 1],
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
                {"type": "experiment-menu-button", "experiment": "Experiment 3"},
            ]
        )
        
        assert result[0] == "Experiment 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Tests for delete confirmation modal button interactions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dash_app import handle_delete_confirmation


class TestDeleteButtonClick:
    """Test that Delete button click is properly detected and handled."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_button_click_triggers_callback(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that clicking Delete button triggers the callback."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,  # Button was clicked
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify deletion occurred
        assert len(result[0]) == 1
        assert result[0][0]["name"] == "Experiment 2"
        assert result[1] == "New Experiment"  # Selection changed
        assert result[2] is None  # Experiment name cleared
        assert result[3] is False  # Modal hidden
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_button_with_multiple_clicks(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that multiple clicks on Delete button still work."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        # Simulate double-click (n_clicks=2)
        result = handle_delete_confirmation(
            confirm_clicks=2,  # Double clicked
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Should still delete
        assert len(result[0]) == 0
        assert result[1] == "New Experiment"
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_callback_receives_correct_trigger_id(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that callback correctly identifies the Delete button as the trigger."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify deletion logic was executed (not cancelled)
        assert len(result[0]) == 0  # Experiment deleted
        assert result[3] is False  # Modal closed
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_button_updates_storage(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that clicking Delete button updates storage."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "alice"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 2",
        )
        
        # Verify save_experiments was called
        mock_save.assert_called_once()
        save_user = mock_save.call_args[0][0]
        save_experiments = mock_save.call_args[0][1]
        
        assert save_user == "alice"
        assert len(save_experiments) == 1
        assert save_experiments[0]["name"] == "Experiment 2"
    
    @patch('dash_app.callback_context')
    def test_cancel_button_does_not_delete(self, mock_ctx):
        """Test that clicking Cancel button does not trigger deletion."""
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        result = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,  # Cancel was clicked
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify no deletion occurred
        assert result[0] is no_update
        assert result[1] is no_update
        assert result[2] is None  # Experiment name cleared
        assert result[3] is False  # Modal hidden


class TestModalButtonIDConsistency:
    """Test that button IDs are consistent between rendering and callback."""
    
    def test_delete_button_id_matches(self):
        """Test that Delete button ID in render matches callback Input ID."""
        from dash_app import render_delete_confirmation
        
        # Render the modal
        modal_children, modal_style = render_delete_confirmation(True, "Test Experiment")
        
        # Convert to string and check for the correct ID
        modal_str = str(modal_children)
        
        # The ID should match exactly what the callback expects
        assert "delete-confirmation-confirm" in modal_str
        assert "delete-confirmation-cancel" in modal_str
    
    def test_button_rendering_structure(self):
        """Test that buttons are rendered with correct structure."""
        from dash_app import render_delete_confirmation
        
        modal_children, modal_style = render_delete_confirmation(True, "Test Experiment")
        
        # Modal should be visible
        assert modal_style == {"display": "block"}
        
        # Modal should contain experiment name
        modal_str = str(modal_children)
        assert "Test Experiment" in modal_str
        
        # Both buttons should be present
        assert "Cancel" in modal_str
        assert "Delete" in modal_str


class TestDeleteButtonWithDifferentStates:
    """Test Delete button behavior with various application states."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_when_experiment_is_selected(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting the currently selected experiment."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",  # Same as being deleted
        )
        
        # Should switch to "New Experiment"
        assert result[1] == "New Experiment"
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_when_different_experiment_selected(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting when a different experiment is selected."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 2",  # Different from being deleted
        )
        
        # Should keep selection on Experiment 2
        assert result[1] == "Experiment 2"
    
    @patch('dash_app.callback_context')
    def test_delete_with_no_experiment_name(self, mock_ctx):
        """Test that Delete button with no experiment name prevents update."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name=None,  # No experiment name
                experiments=[{"name": "Experiment 1"}],
                selected="Experiment 1",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


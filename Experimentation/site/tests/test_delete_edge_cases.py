"""Edge case tests for delete functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dash_app import (
    show_delete_confirmation,
    render_delete_confirmation,
    handle_delete_confirmation,
)


class TestEdgeCases:
    """Test edge cases for delete functionality."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_only_experiment(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test deleting the only experiment in the list."""
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
        
        # Should return empty list
        assert len(result[0]) == 0
        # Should switch to "New Experiment"
        assert result[1] == "New Experiment"
        # Save should be called with empty list
        mock_save.assert_called_once()
        assert len(mock_save.call_args[0][1]) == 0
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_all_experiments_one_by_one(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting all experiments one by one."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        # Delete first
        result1 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        assert len(result1[0]) == 1
        
        # Delete second (last)
        result2 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 2",
            experiments=result1[0],
            selected="Experiment 2",
        )
        
        assert len(result2[0]) == 0
        assert result2[1] == "New Experiment"
    
    @patch('dash_app.callback_context')
    def test_clicking_menu_button_twice_fast(self, mock_ctx):
        """Test double-clicking menu button."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # First click (count = 1)
        result1 = show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        # Second click (count = 2)
        result2 = show_delete_confirmation([2], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        # Both should show modal for same experiment
        assert result1[0] == "Experiment 1"
        assert result2[0] == "Experiment 1"
        assert result1[1] is True
        assert result2[1] is True
    
    @patch('dash_app.callback_context')
    def test_delete_nonexistent_experiment(self, mock_ctx):
        """Test trying to delete experiment that doesn't exist."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        # Try to delete experiment that doesn't exist
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name=None,  # No experiment name
                experiments=experiments,
                selected="Experiment 1",
            )
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_with_empty_experiment_list(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test delete with empty experiment list."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        # Empty experiments list
        experiments = []
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="New Experiment",
        )
        
        # Should return empty list
        assert len(result[0]) == 0
        # Should stay on "New Experiment"
        assert result[1] == "New Experiment"
    
    @patch('dash_app.callback_context')
    def test_clicking_cancel_multiple_times(self, mock_ctx):
        """Test clicking cancel button multiple times."""
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        # First cancel
        result1 = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Second cancel
        result2 = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=2,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Both should close modal without changes
        assert result1[0] is no_update
        assert result1[3] is False
        assert result2[0] is no_update
        assert result2[3] is False
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_with_special_characters_in_name(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting experiment with special characters in name."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment #1 (Test's \"Special\" Chars)", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment #1 (Test's \"Special\" Chars)",
            experiments=experiments,
            selected="Experiment 2",
        )
        
        # Should delete the experiment with special chars
        assert len(result[0]) == 1
        assert result[0][0]["name"] == "Experiment 2"
    
    def test_render_modal_with_long_experiment_name(self):
        """Test rendering modal with very long experiment name."""
        long_name = "A" * 200  # Very long name
        
        modal_result = render_delete_confirmation(True, long_name)
        
        # Should still render modal
        assert modal_result[0] is not None
        assert modal_result[1] == {"display": "block"}
        assert long_name in str(modal_result[0])
    
    def test_render_modal_with_empty_experiment_name(self):
        """Test rendering modal with empty experiment name."""
        modal_result = render_delete_confirmation(True, "")
        
        # Should hide modal
        assert modal_result[0] is None
        assert modal_result[1] == {"display": "none"}
    
    @patch('dash_app.callback_context')
    def test_show_confirmation_with_many_experiments(self, mock_ctx):
        """Test showing confirmation with many experiments in list."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 50"}
        
        # 100 experiments
        clicks = [0] * 100
        clicks[49] = 1  # Experiment 50 clicked
        
        button_ids = [
            {"type": "experiment-menu-button", "experiment": f"Experiment {i+1}"}
            for i in range(100)
        ]
        
        result = show_delete_confirmation(clicks, button_ids)
        
        # Should correctly identify Experiment 50
        assert result[0] == "Experiment 50"
        assert result[1] is True


class TestErrorHandling:
    """Test error handling in delete functionality."""
    
    @patch('dash_app.callback_context')
    def test_invalid_trigger_format(self, mock_ctx):
        """Test handling of invalid trigger format."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "invalid-string"  # Should be dict
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_trigger_missing_required_fields(self, mock_ctx):
        """Test handling of trigger missing required fields."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button"}  # Missing "experiment"
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_unexpected_trigger_id(self, mock_ctx):
        """Test handling of unexpected trigger ID."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "unexpected-id"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=0,
                cancel_clicks=0,
                experiment_name="Experiment 1",
                experiments=[],
                selected="New Experiment",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


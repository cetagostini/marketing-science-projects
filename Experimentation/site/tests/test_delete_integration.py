"""Integration tests for delete functionality workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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


class TestFullDeleteWorkflow:
    """Test complete delete workflow from button click to storage update."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_complete_delete_workflow(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test complete delete workflow: click menu → see modal → click delete → experiment removed."""
        mock_session.update({"user": "test_user"})
        mock_context.return_value = True
        
        # Step 1: Click menu button
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        show_result = show_delete_confirmation(
            [1],
            [{"type": "experiment-menu-button", "experiment": "Experiment 1"}]
        )
        
        assert show_result[0] == "Experiment 1"  # experiment name
        assert show_result[1] is True  # visible
        
        # Step 2: Render modal
        modal_result = render_delete_confirmation(
            visible=show_result[1],
            experiment_name=show_result[0]
        )
        
        assert modal_result[0] is not None  # modal content
        assert modal_result[1] == {"display": "block"}  # modal style
        assert "Experiment 1" in str(modal_result[0])
        
        # Step 3: Confirm deletion
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        delete_result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name=show_result[0],
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify deletion
        assert len(delete_result[0]) == 1  # experiments list
        assert delete_result[0][0]["name"] == "Experiment 2"
        assert delete_result[1] == "New Experiment"  # new selection
        assert delete_result[2] is None  # experiment name cleared
        assert delete_result[3] is False  # modal hidden
        
        # Verify save was called
        mock_save.assert_called_once()
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_cancel_workflow(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test cancel workflow: click menu → see modal → click cancel → experiment remains."""
        from dash import no_update
        
        mock_session.update({"user": "test_user"})
        mock_context.return_value = True
        
        # Step 1: Click menu button
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        show_result = show_delete_confirmation(
            [1],
            [{"type": "experiment-menu-button", "experiment": "Experiment 1"}]
        )
        
        # Step 2: Render modal
        modal_result = render_delete_confirmation(
            visible=show_result[1],
            experiment_name=show_result[0]
        )
        
        assert modal_result[0] is not None
        
        # Step 3: Cancel deletion
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        cancel_result = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,
            experiment_name=show_result[0],
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify nothing deleted
        assert cancel_result[0] is no_update
        assert cancel_result[1] is no_update
        assert cancel_result[2] is None  # experiment name cleared
        assert cancel_result[3] is False  # modal hidden
        
        # Save should NOT be called
        mock_save.assert_not_called()
    
    @patch('dash_app.callback_context')
    def test_different_experiments_show_correct_name(self, mock_ctx):
        """Test clicking menu on different experiments shows correct name in modal."""
        # Test with Experiment 1
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        result1 = show_delete_confirmation(
            [1, 0],
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
            ]
        )
        
        modal1 = render_delete_confirmation(result1[1], result1[0])
        assert "Experiment 1" in str(modal1[0])
        
        # Test with Experiment 2
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 2"}
        result2 = show_delete_confirmation(
            [0, 1],
            [
                {"type": "experiment-menu-button", "experiment": "Experiment 1"},
                {"type": "experiment-menu-button", "experiment": "Experiment 2"},
            ]
        )
        
        modal2 = render_delete_confirmation(result2[1], result2[0])
        assert "Experiment 2" in str(modal2[0])


class TestDeleteStateConsistency:
    """Test state consistency after delete operations."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_state_after_delete(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test state consistency after delete."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete", "message": "Test 1"},
            {"name": "Experiment 2", "status": "complete", "message": "Test 2"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 2",  # Different experiment selected
        )
        
        updated_experiments = result[0]
        new_selection = result[1]
        
        # State should be consistent
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        assert updated_experiments[0]["message"] == "Test 2"
        assert new_selection == "Experiment 2"  # Selection unchanged
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_modal_state_after_cancel(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test modal state is correctly cleared after cancel."""
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        result = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,
            experiment_name="Experiment 1",
            experiments=[{"name": "Experiment 1"}],
            selected="Experiment 1",
        )
        
        # Modal state cleared
        assert result[2] is None  # experiment name
        assert result[3] is False  # visible


class TestMultipleExperimentsDelete:
    """Test deleting multiple experiments in sequence."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_multiple_experiments_sequence(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting multiple experiments one by one."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
            {"name": "Experiment 3", "status": "complete"},
        ]
        
        # Delete first experiment
        result1 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 2",
        )
        
        assert len(result1[0]) == 2
        assert result1[0][0]["name"] == "Experiment 2"
        assert result1[0][1]["name"] == "Experiment 3"
        
        # Delete second experiment from updated list
        result2 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 3",
            experiments=result1[0],
            selected="Experiment 2",
        )
        
        assert len(result2[0]) == 1
        assert result2[0][0]["name"] == "Experiment 2"
        
        # Delete last experiment
        result3 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 2",
            experiments=result2[0],
            selected="Experiment 2",
        )
        
        assert len(result3[0]) == 0
        assert result3[1] == "New Experiment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Comprehensive tests for experiment deletion functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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


class TestShowDeleteConfirmation:
    """Test the show_delete_confirmation callback."""
    
    @patch('dash_app.callback_context')
    def test_show_confirmation_for_valid_experiment(self, mock_ctx):
        """Test that clicking menu button shows confirmation for valid experiment."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # With actual click count > 0
        result = show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
        
        # Should return experiment name and visible=True
        assert result[0] == "Experiment 1"
        assert result[1] is True
    
    @patch('dash_app.callback_context')
    def test_show_confirmation_with_multiple_experiments(self, mock_ctx):
        """Test clicking menu button when multiple experiments exist."""
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 2"}
        
        button_ids = [
            {"type": "experiment-menu-button", "experiment": "Experiment 1"},
            {"type": "experiment-menu-button", "experiment": "Experiment 2"},
            {"type": "experiment-menu-button", "experiment": "Experiment 3"},
        ]
        
        result = show_delete_confirmation([0, 1, 0], button_ids)
        
        # Should return the clicked experiment
        assert result[0] == "Experiment 2"
        assert result[1] is True
    
    @patch('dash_app.callback_context')
    def test_no_trigger_prevents_update(self, mock_ctx):
        """Test that no trigger raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = None
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([0], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_invalid_trigger_prevents_update(self, mock_ctx):
        """Test that invalid trigger (not dict) raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "invalid-string-trigger"
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_missing_experiment_name_prevents_update(self, mock_ctx):
        """Test that trigger without experiment name raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button"}  # Missing "experiment" key
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_zero_clicks_prevents_update(self, mock_ctx):
        """Test that all zero clicks raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        # All clicks are 0 (initial state)
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([0], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])
    
    @patch('dash_app.callback_context')
    def test_wrong_button_type_prevents_update(self, mock_ctx):
        """Test that wrong button type raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = {"type": "experiment-card", "value": "Experiment 1"}  # Wrong type
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation([1], [{"type": "experiment-menu-button", "experiment": "Experiment 1"}])


class TestRenderDeleteConfirmation:
    """Test the render_delete_confirmation callback."""
    
    def test_render_modal_when_visible(self):
        """Test that modal is rendered when visible is True."""
        result = render_delete_confirmation(True, "Experiment 1")
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should return modal content and display block
        assert modal_children is not None
        assert modal_style == {"display": "block"}
    
    def test_modal_contains_experiment_name(self):
        """Test that modal message contains the experiment name."""
        result = render_delete_confirmation(True, "My Test Experiment")
        
        modal_children = result[0]
        
        # Convert to string to check content
        modal_str = str(modal_children)
        assert "My Test Experiment" in modal_str
    
    def test_modal_contains_confirmation_buttons(self):
        """Test that modal contains Cancel and Delete buttons."""
        result = render_delete_confirmation(True, "Experiment 1")
        
        modal_children = result[0]
        
        # Check structure - should have overlay with modal inside
        assert modal_children.className == "delete-confirmation-overlay"
        modal_content = modal_children.children
        assert modal_content.className == "delete-confirmation-modal"
    
    def test_hide_modal_when_not_visible(self):
        """Test that modal is hidden when visible is False."""
        result = render_delete_confirmation(False, "Experiment 1")
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should return None and display none
        assert modal_children is None
        assert modal_style == {"display": "none"}
    
    def test_hide_modal_when_no_experiment_name(self):
        """Test that modal is hidden when experiment name is None."""
        result = render_delete_confirmation(True, None)
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should return None and display none
        assert modal_children is None
        assert modal_style == {"display": "none"}
    
    def test_hide_modal_when_empty_experiment_name(self):
        """Test that modal is hidden when experiment name is empty."""
        result = render_delete_confirmation(True, "")
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should return None and display none
        assert modal_children is None
        assert modal_style == {"display": "none"}


class TestHandleDeleteConfirmation:
    """Test the handle_delete_confirmation callback."""
    
    def _make_session_mock(self, initial_data=None):
        """Create a session mock that behaves like Flask session."""
        data = initial_data or {"user": "test_user"}
        mock = MagicMock()
        mock.get = data.get
        mock.setdefault = data.setdefault
        mock.__getitem__ = data.__getitem__
        mock.__setitem__ = data.__setitem__
        mock.update = data.update
        return mock
    
    @patch('dash_app.callback_context')
    def test_cancel_closes_modal(self, mock_ctx):
        """Test that clicking Cancel closes modal without deleting."""
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Should return no_update for experiments and selection, but close modal
        assert result[0] is no_update
        assert result[1] is no_update
        assert result[2] is None  # Clear experiment name
        assert result[3] is False  # Hide modal
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_confirm_deletes_experiment(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test that clicking Confirm deletes the experiment."""
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
            selected="Experiment 2",  # Different experiment selected
        )
        
        updated_experiments = result[0]
        new_selection = result[1]
        
        # Should delete Experiment 1
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        
        # Should keep current selection (Experiment 2)
        assert new_selection == "Experiment 2"
        
        # Should close modal
        assert result[2] is None
        assert result[3] is False
        
        # Should call save_experiments
        mock_save.assert_called_once()
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_confirm_deletes_selected_experiment_switches_view(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that deleting currently selected experiment switches to New Experiment."""
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
            selected="Experiment 1",  # Same experiment is selected
        )
        
        updated_experiments = result[0]
        new_selection = result[1]
        
        # Should delete Experiment 1
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        
        # Should switch to "New Experiment"
        assert new_selection == "New Experiment"
        
        # Should close modal
        assert result[2] is None
        assert result[3] is False
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_confirm_saves_to_storage(self, mock_ctx, mock_save, mock_context, mock_session):
        """Test that deletion persists to storage."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "alice"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify save_experiments was called with correct user and updated list
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == "alice"
        assert len(call_args[1]) == 1
        assert call_args[1][0]["name"] == "Experiment 2"
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_only_experiment_returns_empty_list(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting the only experiment returns empty list."""
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
        
        updated_experiments = result[0]
        new_selection = result[1]
        
        # Should return empty list
        assert len(updated_experiments) == 0
        
        # Should switch to "New Experiment"
        assert new_selection == "New Experiment"
    
    @patch('dash_app.callback_context')
    def test_no_experiment_name_prevents_update(self, mock_ctx):
        """Test that confirming without experiment name raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name=None,  # No experiment name
                experiments=[],
                selected="New Experiment",
            )
    
    @patch('dash_app.callback_context')
    def test_empty_experiment_name_prevents_update(self, mock_ctx):
        """Test that confirming with empty experiment name raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name="",  # Empty experiment name
                experiments=[],
                selected="New Experiment",
            )
    
    @patch('dash_app.callback_context')
    def test_invalid_trigger_prevents_update(self, mock_ctx):
        """Test that invalid trigger raises PreventUpdate."""
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "invalid-trigger"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=0,
                cancel_clicks=0,
                experiment_name="Experiment 1",
                experiments=[],
                selected="New Experiment",
            )


class TestDeleteStateConsistency:
    """Test state consistency across delete operations."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_multiple_deletes_maintain_consistency(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that multiple deletions maintain state consistency."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        # Start with 3 experiments
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
        assert result1[1] == "Experiment 2"
        
        # Delete another experiment from updated list
        result2 = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 3",
            experiments=result1[0],
            selected="Experiment 2",
        )
        
        assert len(result2[0]) == 1
        assert result2[0][0]["name"] == "Experiment 2"
        assert result2[1] == "Experiment 2"
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_preserves_experiment_data(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that deletion preserves data of remaining experiments."""
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {
                "name": "Experiment 1",
                "status": "complete",
                "message": "Test 1",
                "results": {"summary_mean": {"Impact": {"value": "+10"}}},
            },
            {
                "name": "Experiment 2",
                "status": "complete",
                "message": "Test 2",
                "results": {"summary_mean": {"Impact": {"value": "+20"}}},
            },
        ]
        
        result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 2",
        )
        
        updated_experiments = result[0]
        
        # Verify remaining experiment has all its data
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        assert updated_experiments[0]["message"] == "Test 2"
        assert updated_experiments[0]["results"]["summary_mean"]["Impact"]["value"] == "+20"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


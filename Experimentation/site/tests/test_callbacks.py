"""Unit tests for Dash callback functions with focus on error handling and state consistency."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

import pytest
import pandas as pd

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dash_app import (
    process_experiment,
    add_loading_experiment,
    cleanup_stuck_experiments,
    parse_upload,
    ExperimentFailure,
    EXPERIMENT_TIMEOUT_SECONDS,
)


class TestParseUpload:
    """Test upload parsing functionality."""
    
    def test_parse_upload_success(self):
        """Test successful CSV parsing."""
        csv_content = "date,value\n2025-01-01,100\n2025-01-02,150"
        encoded = "data:text/csv;base64,ZGF0ZSx2YWx1ZQoyMDI1LTAxLTAxLDEwMAoyMDI1LTAxLTAyLDE1MA=="
        
        result = parse_upload(encoded, "test.csv")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "date" in result.columns
        assert "value" in result.columns
    
    def test_parse_upload_no_contents(self):
        """Test parsing with no contents."""
        result = parse_upload(None, "test.csv")
        assert result is None
    
    def test_parse_upload_no_filename(self):
        """Test parsing with no filename."""
        result = parse_upload("data:text/csv;base64,abc", None)
        assert result is None
    
    def test_parse_upload_invalid_base64(self):
        """Test parsing with invalid base64 data."""
        with pytest.raises(ExperimentFailure, match="Unable to decode"):
            parse_upload("data:text/csv;base64,!!invalid!!", "test.csv")
    
    def test_parse_upload_invalid_csv(self):
        """Test parsing with invalid CSV data."""
        # This actually parses as a single-column CSV, so let's test with truly invalid data
        # Using binary data that can't be decoded as UTF-8
        import base64
        invalid_bytes = b'\xff\xfe\x00\x01\x02\x03'  # Invalid UTF-8
        encoded_invalid = "data:text/csv;base64," + base64.b64encode(invalid_bytes).decode('ascii')
        
        with pytest.raises(ExperimentFailure, match="Failed to parse"):
            parse_upload(encoded_invalid, "test.csv")


class TestAddLoadingExperiment:
    """Test the add_loading_experiment callback."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = Mock()
        self.mock_context.triggered_id = "composer-send"
    
    @patch('dash_app.callback_context')
    @patch('dash_app.build_composer_chips')
    @patch('dash_app.build_upload_message')
    def test_add_loading_experiment_success(self, mock_upload_msg, mock_chips, mock_ctx):
        """Test successfully adding a loading experiment."""
        mock_ctx.triggered_id = "composer-send"
        mock_chips.return_value = []
        mock_upload_msg.return_value = None
        
        experiments = []
        message = "Test experiment"
        upload_contents = "data:text/csv;base64,abc"
        upload_filename = "test.csv"
        processing_metadata = {}
        
        result = add_loading_experiment(
            send_clicks=1,
            experiments=experiments,
            message=message,
            upload_contents=upload_contents,
            upload_filename=upload_filename,
            current_class="attachment-popover",
            processing_metadata=processing_metadata,
        )
        
        # Check that result has correct structure
        assert len(result) == 13  # All outputs including processing_metadata
        updated_experiments = result[0]
        processing_data = result[1]
        updated_metadata = result[12]
        
        # Verify experiment was added
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["status"] == "loading"
        assert updated_experiments[0]["name"] == "Experiment 1"
        
        # Verify processing data was created
        assert processing_data is not None
        assert processing_data["experiment_name"] == "Experiment 1"
        assert processing_data["message"] == message
        
        # Verify metadata was updated
        assert "Experiment 1" in updated_metadata
        assert updated_metadata["Experiment 1"]["status"] == "processing"
        assert "start_time" in updated_metadata["Experiment 1"]
    
    @patch('dash_app.callback_context')
    def test_add_loading_experiment_empty_message(self, mock_ctx):
        """Test adding experiment with empty message fails."""
        mock_ctx.triggered_id = "composer-send"
        
        result = add_loading_experiment(
            send_clicks=1,
            experiments=[],
            message="   ",  # Empty after strip
            upload_contents="data:text/csv;base64,abc",
            upload_filename="test.csv",
            current_class="attachment-popover",
            processing_metadata={},
        )
        
        # Should return error without updating
        error_notification = result[10]
        assert error_notification is not None
        assert "cannot be empty" in str(error_notification)
    
    @patch('dash_app.callback_context')
    def test_add_loading_experiment_no_dataset(self, mock_ctx):
        """Test adding experiment without dataset fails."""
        mock_ctx.triggered_id = "composer-send"
        
        result = add_loading_experiment(
            send_clicks=1,
            experiments=[],
            message="Test",
            upload_contents=None,
            upload_filename=None,
            current_class="attachment-popover",
            processing_metadata={},
        )
        
        # Should return error
        error_notification = result[10]
        assert error_notification is not None
        assert "attach a dataset" in str(error_notification)


class TestProcessExperiment:
    """Test the process_experiment callback with various scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Session only needs user identifier (experiments are in file storage)
        self.mock_session = {"user": "test_user"}
        self.sample_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=30),
            "DEU": range(100, 130),
            "SWE": range(90, 120),
            "NOR": range(95, 125),
        })
    
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
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    @patch('dash_app.current_app')
    @patch('dash_app.run_experiment_pipeline')
    def test_process_experiment_success(
        self, mock_pipeline, mock_app, mock_parse, mock_save, mock_session
    ):
        """Test successful experiment processing."""
        # Setup mocks
        mock_session.update({"user": "test_user", "experiments": {}})
        mock_parse.return_value = self.sample_df
        mock_app.config.get.return_value = Mock()  # Mock OpenAI client
        
        mock_pipeline.return_value = {
            "intervention_start_date": "2025-01-15",
            "intervention_end_date": "2025-01-30",
            "target_var": "DEU",
            "control_group": ["SWE", "NOR"],
            "series": {},
            "summary_mean": {},
            "summary_cumulative": {},
            "lift_table": [],
        }
        
        processing_data = {
            "experiment_name": "Experiment 1",
            "message": "Test experiment",
            "upload_contents": "data:text/csv;base64,abc",
            "upload_filename": "test.csv",
        }
        
        experiments = [{"name": "Experiment 1", "status": "loading"}]
        processing_metadata = {
            "Experiment 1": {
                "start_time": datetime.now().isoformat(),
                "status": "processing",
            }
        }
        
        result = process_experiment(processing_data, experiments, processing_metadata)
        
        # Verify result structure (7 outputs)
        assert len(result) == 7
        updated_experiments = result[0]
        selected_experiment = result[1]
        updated_metadata = result[6]
        
        # Verify experiment was completed
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["status"] == "complete"
        assert updated_experiments[0]["name"] == "Experiment 1"
        
        # Verify selection
        assert selected_experiment == "Experiment 1"
        
        # Verify metadata updated
        assert updated_metadata["Experiment 1"]["status"] == "complete"
        
        # Verify save was called
        mock_save.assert_called_once()
    
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    def test_process_experiment_parse_failure(self, mock_parse, mock_save):
        """Test experiment processing when parsing fails."""
        # Create proper session mock
        with patch('dash_app.session', self._make_session_mock({"user": "test_user"})):
            mock_parse.return_value = None
            
            processing_data = {
                "experiment_name": "Experiment 1",
                "message": "Test",
                "upload_contents": "data:text/csv;base64,abc",
                "upload_filename": "test.csv",
            }
            
            experiments = [{"name": "Experiment 1", "status": "loading"}]
            processing_metadata = {"Experiment 1": {"start_time": datetime.now().isoformat()}}
            
            result = process_experiment(processing_data, experiments, processing_metadata)
            
            # Verify cleanup happened
            updated_experiments = result[0]
            selected_experiment = result[1]
            error_msg = result[4]
            
            # Loading experiment should be removed
            assert len(updated_experiments) == 0
            
            # Should go back to new experiment view
            assert selected_experiment == "New Experiment"
            
            # Should have error message
            assert error_msg is not None
            
            # Save should still be called (cleanup)
            mock_save.assert_called_once()
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    @patch('dash_app.current_app')
    def test_process_experiment_no_openai_client(
        self, mock_app, mock_parse, mock_save, mock_session
    ):
        """Test experiment processing when OpenAI client is not configured."""
        mock_session.update({"user": "test_user", "experiments": {}})
        mock_parse.return_value = self.sample_df
        mock_app.config.get.return_value = None  # No OpenAI client
        
        processing_data = {
            "experiment_name": "Experiment 1",
            "message": "Test",
            "upload_contents": "data:text/csv;base64,abc",
            "upload_filename": "test.csv",
        }
        
        experiments = [{"name": "Experiment 1", "status": "loading"}]
        processing_metadata = {"Experiment 1": {"start_time": datetime.now().isoformat()}}
        
        result = process_experiment(processing_data, experiments, processing_metadata)
        
        # Verify error handling
        updated_experiments = result[0]
        error_msg = result[4]
        
        # Loading experiment should be removed
        assert len(updated_experiments) == 0
        
        # Should have error about OpenAI
        assert error_msg is not None
        assert "OpenAI" in str(error_msg) or "not configured" in str(error_msg)
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    @patch('dash_app.current_app')
    @patch('dash_app.run_experiment_pipeline')
    def test_process_experiment_pipeline_exception(
        self, mock_pipeline, mock_app, mock_parse, mock_save, mock_session
    ):
        """Test experiment processing when pipeline raises exception."""
        mock_session.update({"user": "test_user", "experiments": {}})
        mock_parse.return_value = self.sample_df
        mock_app.config.get.return_value = Mock()
        mock_pipeline.side_effect = Exception("Pipeline error")
        
        processing_data = {
            "experiment_name": "Experiment 1",
            "message": "Test",
            "upload_contents": "data:text/csv;base64,abc",
            "upload_filename": "test.csv",
        }
        
        experiments = [{"name": "Experiment 1", "status": "loading"}]
        processing_metadata = {"Experiment 1": {"start_time": datetime.now().isoformat()}}
        
        result = process_experiment(processing_data, experiments, processing_metadata)
        
        # Verify cleanup happened
        updated_experiments = result[0]
        error_msg = result[4]
        updated_metadata = result[6]
        
        # Loading experiment should be removed
        assert len(updated_experiments) == 0
        
        # Should have error message
        assert error_msg is not None
        
        # Metadata should be updated
        assert updated_metadata["Experiment 1"]["status"] == "complete"
    
    def test_process_experiment_no_processing_data(self):
        """Test that callback prevents update when no processing data."""
        from dash.exceptions import PreventUpdate
        
        with pytest.raises(PreventUpdate):
            process_experiment(None, [], {})


class TestCleanupStuckExperiments:
    """Test the cleanup callback for stuck experiments."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.has_request_context')
    def test_cleanup_stuck_experiments(self, mock_context, mock_save, mock_session):
        """Test cleanup of experiments that exceed timeout."""
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        # Create an experiment that started too long ago
        old_time = datetime.now()
        # Subtract more than timeout
        old_time = old_time.replace(
            year=old_time.year - 1
        )  # Definitely exceeds timeout
        
        experiments = [
            {"name": "Experiment 1", "status": "loading"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        processing_metadata = {
            "Experiment 1": {
                "start_time": old_time.isoformat(),
                "status": "processing",
            }
        }
        
        result = cleanup_stuck_experiments(1, experiments, processing_metadata)
        
        updated_experiments = result[0]
        updated_metadata = result[1]
        
        # Stuck experiment should be removed
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        
        # Metadata should show timeout
        assert updated_metadata["Experiment 1"]["status"] == "timeout"
        
        # Save should be called
        mock_save.assert_called_once()
    
    def test_cleanup_no_stuck_experiments(self):
        """Test that cleanup doesn't run when no experiments are stuck."""
        from dash.exceptions import PreventUpdate
        
        # Recent time
        recent_time = datetime.now()
        
        experiments = [{"name": "Experiment 1", "status": "loading"}]
        processing_metadata = {
            "Experiment 1": {
                "start_time": recent_time.isoformat(),
                "status": "processing",
            }
        }
        
        with pytest.raises(PreventUpdate):
            cleanup_stuck_experiments(1, experiments, processing_metadata)
    
    def test_cleanup_no_experiments(self):
        """Test cleanup with no experiments."""
        from dash.exceptions import PreventUpdate
        
        with pytest.raises(PreventUpdate):
            cleanup_stuck_experiments(1, [], {})
    
    def test_cleanup_no_metadata(self):
        """Test cleanup with no metadata."""
        from dash.exceptions import PreventUpdate
        
        experiments = [{"name": "Experiment 1", "status": "loading"}]
        
        with pytest.raises(PreventUpdate):
            cleanup_stuck_experiments(1, experiments, {})


class TestStateConsistency:
    """Test state consistency across callbacks."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    @patch('dash_app.current_app')
    @patch('dash_app.run_experiment_pipeline')
    def test_experiments_always_saved_on_completion(
        self, mock_pipeline, mock_app, mock_parse, mock_save, mock_session
    ):
        """Test that experiments are always saved, even on error."""
        mock_session.update({"user": "test_user"})
        mock_parse.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_app.config.get.return_value = Mock()
        mock_pipeline.side_effect = Exception("Error")
        
        processing_data = {
            "experiment_name": "Test",
            "message": "Test",
            "upload_contents": "data:text/csv;base64,abc",
            "upload_filename": "test.csv",
        }
        
        experiments = [{"name": "Test", "status": "loading"}]
        processing_metadata = {"Test": {"start_time": datetime.now().isoformat()}}
        
        # Run callback (will error but should save)
        result = process_experiment(processing_data, experiments, processing_metadata)
        
        # Verify save was called (cleanup)
        assert mock_save.called
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.save_experiments')
    @patch('dash_app.parse_upload')
    @patch('dash_app.current_app')
    @patch('dash_app.run_experiment_pipeline')
    def test_metadata_always_updated(
        self, mock_pipeline, mock_app, mock_parse, mock_save, mock_session
    ):
        """Test that metadata is always updated, even on error."""
        mock_session.update({"user": "test_user"})
        mock_parse.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_app.config.get.return_value = Mock()
        mock_pipeline.side_effect = Exception("Error")
        
        processing_data = {
            "experiment_name": "Test",
            "message": "Test",
            "upload_contents": "data:text/csv;base64,abc",
            "upload_filename": "test.csv",
        }
        
        experiments = [{"name": "Test", "status": "loading"}]
        processing_metadata = {"Test": {"start_time": datetime.now().isoformat()}}
        
        result = process_experiment(processing_data, experiments, processing_metadata)
        
        updated_metadata = result[6]
        
        # Metadata should show completion
        assert updated_metadata["Test"]["status"] == "complete"
        assert "end_time" in updated_metadata["Test"]


class TestDeleteCallbacks:
    """Test delete experiment callbacks."""
    
    @patch('dash_app.callback_context')
    def test_show_delete_confirmation_callback(self, mock_ctx):
        """Test show_delete_confirmation callback with valid input."""
        from dash_app import show_delete_confirmation
        
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        result = show_delete_confirmation(
            clicks=[1],
            button_ids=[{"type": "experiment-menu-button", "experiment": "Experiment 1"}]
        )
        
        # Should return experiment name and visible=True
        assert result == ("Experiment 1", True)
    
    @patch('dash_app.callback_context')
    def test_show_delete_confirmation_prevents_update_no_trigger(self, mock_ctx):
        """Test show_delete_confirmation raises PreventUpdate when no trigger."""
        from dash_app import show_delete_confirmation
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = None
        
        with pytest.raises(PreventUpdate):
            show_delete_confirmation(
                clicks=[1],
                button_ids=[{"type": "experiment-menu-button", "experiment": "Experiment 1"}]
            )
    
    def test_render_delete_confirmation_visible(self):
        """Test render_delete_confirmation renders modal when visible."""
        from dash_app import render_delete_confirmation
        
        result = render_delete_confirmation(visible=True, experiment_name="Test Experiment")
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should render modal
        assert modal_children is not None
        assert modal_style == {"display": "block"}
        
        # Check modal contains experiment name
        modal_str = str(modal_children)
        assert "Test Experiment" in modal_str
    
    def test_render_delete_confirmation_hidden(self):
        """Test render_delete_confirmation hides modal when not visible."""
        from dash_app import render_delete_confirmation
        
        result = render_delete_confirmation(visible=False, experiment_name="Test Experiment")
        
        modal_children = result[0]
        modal_style = result[1]
        
        # Should hide modal
        assert modal_children is None
        assert modal_style == {"display": "none"}
    
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_cancel(self, mock_ctx):
        """Test handle_delete_confirmation with cancel action."""
        from dash_app import handle_delete_confirmation
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        result = handle_delete_confirmation(
            confirm_clicks=0,
            cancel_clicks=1,
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Should not update experiments or selection, but close modal
        assert result[0] is no_update
        assert result[1] is no_update
        assert result[2] is None  # Clear experiment name
        assert result[3] is False  # Hide modal
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_confirm(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test handle_delete_confirmation with confirm action."""
        from dash_app import handle_delete_confirmation
        
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
            selected="Experiment 2",
        )
        
        updated_experiments = result[0]
        new_selection = result[1]
        
        # Should delete experiment
        assert len(updated_experiments) == 1
        assert updated_experiments[0]["name"] == "Experiment 2"
        
        # Should keep current selection
        assert new_selection == "Experiment 2"
        
        # Should close modal
        assert result[2] is None
        assert result[3] is False
        
        # Should save
        mock_save.assert_called_once()
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_deletes_selected(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test deleting currently selected experiment switches to New Experiment."""
        from dash_app import handle_delete_confirmation
        
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
            selected="Experiment 1",  # Same as deleted
        )
        
        new_selection = result[1]
        
        # Should switch to New Experiment
        assert new_selection == "New Experiment"
    
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_prevents_update_no_name(self, mock_ctx):
        """Test handle_delete_confirmation raises PreventUpdate with no experiment name."""
        from dash_app import handle_delete_confirmation
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name=None,
                experiments=[],
                selected="New Experiment",
            )
    
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_prevents_update_on_modal_render(self, mock_ctx):
        """Test that callback prevents update when modal first renders (both n_clicks=0)."""
        from dash_app import handle_delete_confirmation
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        # When modal first renders, both buttons have n_clicks=0
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=0,
                cancel_clicks=0,
                experiment_name="Experiment 1",
                experiments=[{"name": "Experiment 1"}],
                selected="New Experiment",
            )
    
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_prevents_update_when_both_none(self, mock_ctx):
        """Test that callback prevents update when both n_clicks are None."""
        from dash_app import handle_delete_confirmation
        from dash.exceptions import PreventUpdate
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        
        # When buttons are None (initial state)
        with pytest.raises(PreventUpdate):
            handle_delete_confirmation(
                confirm_clicks=None,
                cancel_clicks=None,
                experiment_name="Experiment 1",
                experiments=[{"name": "Experiment 1"}],
                selected="New Experiment",
            )
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_works_after_modal_render(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test that delete works when going from n_clicks=0 to n_clicks=1."""
        from dash_app import handle_delete_confirmation
        
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        # First click (n_clicks goes from 0 to 1)
        result = handle_delete_confirmation(
            confirm_clicks=1,  # First actual click
            cancel_clicks=0,   # Not clicked
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify deletion occurred
        assert len(result[0]) == 1
        assert result[0][0]["name"] == "Experiment 2"
        assert result[1] == "New Experiment"
        assert result[2] is None
        assert result[3] is False
    
    @patch('dash_app.callback_context')
    def test_handle_delete_confirmation_cancel_works_with_n_clicks_1(self, mock_ctx):
        """Test that cancel button works when clicked (n_clicks=1)."""
        from dash_app import handle_delete_confirmation
        from dash import no_update
        
        mock_ctx.triggered_id = "delete-confirmation-cancel"
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        result = handle_delete_confirmation(
            confirm_clicks=0,   # Not clicked
            cancel_clicks=1,    # First click
            experiment_name="Experiment 1",
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify cancellation logic
        assert result[0] is no_update
        assert result[1] is no_update
        assert result[2] is None
        assert result[3] is False


class TestDeleteIntegration:
    """Integration tests for delete functionality."""
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    @patch('dash_app.callback_context')
    def test_delete_workflow_end_to_end(
        self, mock_ctx, mock_save, mock_context, mock_session
    ):
        """Test complete delete workflow from button click to storage."""
        from dash_app import show_delete_confirmation, handle_delete_confirmation
        
        # Step 1: Click delete button
        mock_ctx.triggered_id = {"type": "experiment-menu-button", "experiment": "Experiment 1"}
        
        show_result = show_delete_confirmation(
            clicks=[1],
            button_ids=[{"type": "experiment-menu-button", "experiment": "Experiment 1"}]
        )
        
        assert show_result[0] == "Experiment 1"
        assert show_result[1] is True
        
        # Step 2: Confirm deletion
        mock_ctx.triggered_id = "delete-confirmation-confirm"
        mock_context.return_value = True
        mock_session.update({"user": "test_user"})
        
        experiments = [
            {"name": "Experiment 1", "status": "complete"},
            {"name": "Experiment 2", "status": "complete"},
        ]
        
        delete_result = handle_delete_confirmation(
            confirm_clicks=1,
            cancel_clicks=0,
            experiment_name=show_result[0],  # Use result from step 1
            experiments=experiments,
            selected="Experiment 1",
        )
        
        # Verify deletion completed
        assert len(delete_result[0]) == 1
        assert delete_result[0][0]["name"] == "Experiment 2"
        assert delete_result[1] == "New Experiment"
        assert delete_result[2] is None
        assert delete_result[3] is False
        
        # Verify save was called
        mock_save.assert_called_once()
    
    @patch('dash_app.session', new_callable=dict)
    @patch('dash_app.has_request_context')
    @patch('dash_app.save_experiments')
    def test_delete_persists_to_correct_user(self, mock_save, mock_context, mock_session):
        """Test that deletion saves to the correct user's storage."""
        from dash_app import handle_delete_confirmation
        
        mock_context.return_value = True
        mock_session.update({"user": "alice"})
        
        experiments = [{"name": "Experiment 1", "status": "complete"}]
        
        with patch('dash_app.callback_context') as mock_ctx:
            mock_ctx.triggered_id = "delete-confirmation-confirm"
            
            handle_delete_confirmation(
                confirm_clicks=1,
                cancel_clicks=0,
                experiment_name="Experiment 1",
                experiments=experiments,
                selected="Experiment 1",
            )
        
        # Verify save was called with correct user
        mock_save.assert_called_once()
        call_user = mock_save.call_args[0][0]
        assert call_user == "alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Integration tests for planning session variable shadowing fix.

This test verifies that the handle_planning_message callback correctly uses
the Flask session object and doesn't have variable shadowing issues.

To run these tests:
    conda activate synth_experiments
    cd /Users/carlostrujillo/Documents/GitHub/marketing-science-projects/Experimentation/site
    python -m pytest tests/test_planning_session_variable_fix.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Try to import dash and dash_app
try:
    import dash
    from dash.exceptions import PreventUpdate
    from flask import Flask
    import dash_app
    from dash_app import handle_planning_message
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


if not DASH_AVAILABLE:
    pytest.skip("Dash not available - run in synth_experiments conda env", allow_module_level=True)


@pytest.fixture
def mock_flask_app():
    """Create a mock Flask app with proper context."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "test-secret-key"
    return app


class TestPlanningSessionVariableFix:
    """Test that handle_planning_message doesn't have session variable shadowing."""
    
    def test_handle_planning_message_new_session_saves_to_storage(self, mock_flask_app):
        """Test that new planning sessions are saved with correct user key from Flask session."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock validator response
        mock_validator_response = Mock()
        mock_validator_response.output_text = '{"should_continue": true, "reason": ""}'
        mock_client.responses.parse.return_value = mock_validator_response
        
        # Mock chat service response
        mock_chat_response = Mock()
        mock_chat_response.output_text = "Here's my advice on experiment design..."
        mock_client.responses.create.return_value = mock_chat_response
        
        with mock_flask_app.app_context():
            with mock_flask_app.test_request_context():
                from flask import session
                session['user'] = 'test_user'
                
                with patch('dash_app.save_planning_sessions') as mock_save:
                    # Call the callback
                    (
                        updated_sessions,
                        selected_session,
                        updated_messages,
                        input_value,
                        error
                    ) = handle_planning_message(
                        n_clicks=1,
                        user_input="How do I choose controls?",
                        messages=[],
                        selected_session="New Planning",
                        sessions=[]
                    )
                    
                    # Verify no error occurred
                    assert error is None, "Should not have any validation or execution error"
                    
                    # Verify session was saved with correct user key
                    mock_save.assert_called_once()
                    call_args = mock_save.call_args
                    assert call_args[0][0] == 'test_user', "Should use Flask session user key"
                    
                    # Verify new session was created
                    saved_sessions = call_args[0][1]
                    assert len(saved_sessions) == 1
                    assert saved_sessions[0]["name"] == "Planning Session 1"
                    assert len(saved_sessions[0]["messages"]) == 2  # user + assistant
                    
                    # Verify input was cleared
                    assert input_value == ""
    
    def test_handle_planning_message_update_existing_session(self, mock_flask_app):
        """Test that existing planning sessions are updated without variable shadowing errors."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock validator response
        mock_validator_response = Mock()
        mock_validator_response.output_text = '{"should_continue": true, "reason": ""}'
        mock_client.responses.parse.return_value = mock_validator_response
        
        # Mock chat service response
        mock_chat_response = Mock()
        mock_chat_response.output_text = "Follow-up advice..."
        mock_client.responses.create.return_value = mock_chat_response
        
        # Existing sessions - this is the critical test for variable shadowing
        existing_sessions = [
            {
                "name": "Planning Session 1",
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"}
                ],
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            },
            {
                "name": "Planning Session 2",
                "messages": [],
                "created_at": "2024-01-02T00:00:00",
                "updated_at": "2024-01-02T00:00:00"
            }
        ]
        
        with mock_flask_app.app_context():
            with mock_flask_app.test_request_context():
                from flask import session
                session['user'] = 'alice'
                
                with patch('dash_app.save_planning_sessions') as mock_save:
                    # Call the callback - update Planning Session 1
                    (
                        updated_sessions,
                        selected_session,
                        updated_messages,
                        input_value,
                        error
                    ) = handle_planning_message(
                        n_clicks=1,
                        user_input="Follow-up question",
                        messages=[
                            {"role": "user", "content": "First question"},
                            {"role": "assistant", "content": "First answer"}
                        ],
                        selected_session="Planning Session 1",
                        sessions=existing_sessions
                    )
                    
                    # Verify no error occurred (especially no UnboundLocalError)
                    assert error is None, "Should not have variable shadowing error"
                    
                    # Verify session was saved with correct user key from Flask session
                    mock_save.assert_called_once()
                    call_args = mock_save.call_args
                    assert call_args[0][0] == 'alice', "Should use Flask session user key 'alice'"
                    
                    # Verify sessions were updated correctly
                    saved_sessions = call_args[0][1]
                    assert len(saved_sessions) == 2, "Should still have 2 sessions"
                    
                    # Find the updated session
                    updated_session_1 = next(s for s in saved_sessions if s["name"] == "Planning Session 1")
                    assert len(updated_session_1["messages"]) == 4  # 2 old + 2 new
                    
                    # Verify the other session wasn't modified
                    session_2 = next(s for s in saved_sessions if s["name"] == "Planning Session 2")
                    assert len(session_2["messages"]) == 0
    
    def test_flask_session_not_shadowed_by_loop_variable(self, mock_flask_app):
        """Test that Flask session is accessible even when iterating over planning sessions."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock validator response
        mock_validator_response = Mock()
        mock_validator_response.output_text = '{"should_continue": true, "reason": ""}'
        mock_client.responses.parse.return_value = mock_validator_response
        
        # Mock chat service response
        mock_chat_response = Mock()
        mock_chat_response.output_text = "Response text"
        mock_client.responses.create.return_value = mock_chat_response
        
        # Create many existing sessions to ensure loop iteration doesn't shadow Flask session
        many_sessions = [
            {
                "name": f"Planning Session {i}",
                "messages": [{"role": "user", "content": f"Question {i}"}],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            for i in range(1, 6)  # 5 sessions
        ]
        
        with mock_flask_app.app_context():
            with mock_flask_app.test_request_context():
                from flask import session
                session['user'] = 'bob'
                
                with patch('dash_app.save_planning_sessions') as mock_save:
                    # Call the callback to update session in the middle
                    try:
                        (
                            updated_sessions,
                            selected_session,
                            updated_messages,
                            input_value,
                            error
                        ) = handle_planning_message(
                            n_clicks=1,
                            user_input="Update middle session",
                            messages=[{"role": "user", "content": "Question 3"}],
                            selected_session="Planning Session 3",
                            sessions=many_sessions
                        )
                        
                        # If we get here without UnboundLocalError, the fix worked
                        assert error is None
                        
                        # Verify Flask session was used correctly
                        mock_save.assert_called_once()
                        call_args = mock_save.call_args
                        assert call_args[0][0] == 'bob', \
                            "Should access Flask session.get('user', 'default') without UnboundLocalError"
                        
                    except UnboundLocalError as e:
                        pytest.fail(f"UnboundLocalError should not occur: {e}")
    
    def test_empty_input_prevents_update(self, mock_flask_app):
        """Test that empty input prevents callback execution."""
        with mock_flask_app.app_context():
            with pytest.raises(PreventUpdate):
                handle_planning_message(
                    n_clicks=1,
                    user_input="",
                    messages=[],
                    selected_session="New Planning",
                    sessions=[]
                )
    
    def test_no_clicks_prevents_update(self, mock_flask_app):
        """Test that no clicks prevents callback execution."""
        with mock_flask_app.app_context():
            with pytest.raises(PreventUpdate):
                handle_planning_message(
                    n_clicks=0,
                    user_input="Test question",
                    messages=[],
                    selected_session="New Planning",
                    sessions=[]
                )
    
    def test_validation_error_keeps_input(self, mock_flask_app):
        """Test that validation errors keep the input and don't save."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_flask_app.config["OPENAI_CLIENT"] = mock_client
        
        # Mock validator response - reject the question
        mock_validator_response = Mock()
        mock_validator_response.output_text = '{"should_continue": false, "reason": "Off-topic question"}'
        mock_client.responses.parse.return_value = mock_validator_response
        
        with mock_flask_app.app_context():
            with mock_flask_app.test_request_context():
                from flask import session
                session['user'] = 'test_user'
                
                with patch('dash_app.save_planning_sessions') as mock_save:
                    # Call the callback
                    (
                        updated_sessions,
                        selected_session,
                        updated_messages,
                        input_value,
                        error
                    ) = handle_planning_message(
                        n_clicks=1,
                        user_input="What's the weather?",
                        messages=[],
                        selected_session="New Planning",
                        sessions=[]
                    )
                    
                    # Verify error was returned
                    assert error == "Off-topic question"
                    
                    # Verify input was NOT cleared
                    assert input_value == "What's the weather?"
                    
                    # Verify nothing was saved
                    mock_save.assert_not_called()

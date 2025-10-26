"""Integration tests for planning mode (requires dash environment).

These tests import dash_app directly and test actual callback functionality.
They only run when the synth_experiments conda environment is available.

To run these tests:
    conda activate synth_experiments
    cd /Users/carlostrujillo/Documents/GitHub/marketing-science-projects/Experimentation/site
    python -m pytest tests/test_planning_integration.py -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Try to import dash and dash_app
try:
    import dash
    from dash import no_update
    import dash_app
    from dash_app import (
        _generate_planning_session_name,
        _initial_planning_sessions_state,
        switch_mode,
        select_planning_session,
        load_planning_messages,
        render_planning_messages,
        show_delete_planning_confirmation,
    )
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestPlanningHelperFunctions(unittest.TestCase):
    """Test planning helper functions from dash_app."""
    
    def test_generate_planning_session_name(self):
        """Test planning session name generation."""
        name1 = _generate_planning_session_name(1)
        name2 = _generate_planning_session_name(5)
        name10 = _generate_planning_session_name(10)
        
        self.assertEqual(name1, "Planning Session 1")
        self.assertEqual(name2, "Planning Session 5")
        self.assertEqual(name10, "Planning Session 10")
    
    @patch('dash_app.has_request_context')
    def test_initial_planning_sessions_state_no_context(self, mock_has_context):
        """Test initial state with no request context."""
        mock_has_context.return_value = False
        sessions = _initial_planning_sessions_state()
        self.assertEqual(sessions, [])
    
    @patch('dash_app.has_request_context')
    @patch('dash_app.load_planning_sessions')
    @patch('dash_app.session', {'user': 'test_user'})
    def test_initial_planning_sessions_state_with_context(self, mock_load, mock_has_context):
        """Test initial state with request context."""
        mock_has_context.return_value = True
        mock_load.return_value = [
            {"name": "Planning Session 1", "messages": []},
            {"name": "Planning Session 2", "messages": []},
        ]
        
        sessions = _initial_planning_sessions_state()
        
        self.assertEqual(len(sessions), 2)
        mock_load.assert_called_with('test_user')


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestModeSwitchingCallback(unittest.TestCase):
    """Test mode switching callback."""
    
    @patch('dash_app.callback_context')
    def test_switch_to_experiments_mode(self, mock_ctx):
        """Test switching to experiments mode."""
        mock_ctx.triggered_id = "mode-experiments"
        
        mode, exp_class, plan_class = switch_mode(1, 0, "planning")
        
        self.assertEqual(mode, "experiments")
        self.assertIn("active", exp_class)
        self.assertNotIn("active", plan_class)
    
    @patch('dash_app.callback_context')
    def test_switch_to_planning_mode(self, mock_ctx):
        """Test switching to planning mode."""
        mock_ctx.triggered_id = "mode-planning"
        
        mode, exp_class, plan_class = switch_mode(1, 1, "experiments")
        
        self.assertEqual(mode, "planning")
        self.assertNotIn("active", exp_class)
        self.assertIn("active", plan_class)
    
    @patch('dash_app.callback_context')
    def test_initial_load_experiments_mode(self, mock_ctx):
        """Test initial load defaults to experiments mode."""
        mock_ctx.triggered_id = None
        
        mode, exp_class, plan_class = switch_mode(0, 0, "experiments")
        
        self.assertEqual(mode, "experiments")
        self.assertIn("active", exp_class)
    
    @patch('dash_app.callback_context')
    def test_initial_load_planning_mode(self, mock_ctx):
        """Test initial load preserves planning mode if set."""
        mock_ctx.triggered_id = None
        
        mode, exp_class, plan_class = switch_mode(0, 0, "planning")
        
        self.assertEqual(mode, "planning")
        self.assertIn("active", plan_class)


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestPlanningSessionSelection(unittest.TestCase):
    """Test planning session selection callback."""
    
    @patch('dash_app.callback_context')
    def test_select_new_planning(self, mock_ctx):
        """Test selecting 'New Planning'."""
        mock_ctx.triggered_id = {"type": "planning-card", "value": "New Planning"}
        
        result = select_planning_session([1], [{"type": "planning-card", "value": "New Planning"}])
        
        self.assertEqual(result, "New Planning")
    
    @patch('dash_app.callback_context')
    def test_select_existing_session(self, mock_ctx):
        """Test selecting an existing planning session."""
        mock_ctx.triggered_id = {"type": "planning-card", "value": "Planning Session 1"}
        
        result = select_planning_session([0, 1], [
            {"type": "planning-card", "value": "New Planning"},
            {"type": "planning-card", "value": "Planning Session 1"}
        ])
        
        self.assertEqual(result, "Planning Session 1")


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestLoadPlanningMessages(unittest.TestCase):
    """Test loading planning messages callback."""
    
    def test_load_messages_for_new_planning(self):
        """Test loading messages for 'New Planning' returns empty."""
        messages = load_planning_messages("New Planning", [])
        self.assertEqual(messages, [])
    
    def test_load_messages_for_existing_session(self):
        """Test loading messages for existing session."""
        sessions = [
            {
                "name": "Planning Session 1",
                "messages": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"}
                ]
            }
        ]
        
        messages = load_planning_messages("Planning Session 1", sessions)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
    
    def test_load_messages_for_nonexistent_session(self):
        """Test loading messages for non-existent session returns empty."""
        sessions = [{"name": "Planning Session 1", "messages": []}]
        
        messages = load_planning_messages("Planning Session 999", sessions)
        
        self.assertEqual(messages, [])


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestRenderPlanningMessages(unittest.TestCase):
    """Test rendering planning messages callback."""
    
    def test_render_empty_messages(self):
        """Test rendering empty messages shows empty state."""
        result = render_planning_messages([])
        
        # Result should be a Div with empty state content
        self.assertIsNotNone(result)
        # Check if it's a Div (has style property)
        self.assertTrue(hasattr(result, 'style'))
    
    def test_render_user_message(self):
        """Test rendering a user message."""
        messages = [{"role": "user", "content": "Test question"}]
        
        result = render_planning_messages(messages)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
    
    def test_render_conversation(self):
        """Test rendering a full conversation."""
        messages = [
            {"role": "user", "content": "What is Synthetic Control?"},
            {"role": "assistant", "content": "Synthetic Control is a method..."},
            {"role": "user", "content": "When should I use it?"},
            {"role": "assistant", "content": "You should use it when..."},
        ]
        
        result = render_planning_messages(messages)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestDeletePlanningConfirmation(unittest.TestCase):
    """Test planning session deletion confirmation callback."""
    
    @patch('dash_app.callback_context')
    def test_show_delete_confirmation(self, mock_ctx):
        """Test showing delete confirmation for planning session."""
        mock_ctx.triggered_id = {"type": "planning-menu-button", "session": "Planning Session 1"}
        
        from dash.exceptions import PreventUpdate
        
        # This should return session name and True
        session_name, visible = show_delete_planning_confirmation([1], [
            {"type": "planning-menu-button", "session": "Planning Session 1"}
        ])
        
        self.assertEqual(session_name, "Planning Session 1")
        self.assertTrue(visible)
    
    @patch('dash_app.callback_context')
    def test_show_delete_confirmation_wrong_button_type(self, mock_ctx):
        """Test that wrong button type prevents update."""
        mock_ctx.triggered_id = {"type": "wrong-button", "session": "Planning Session 1"}
        
        from dash.exceptions import PreventUpdate
        
        with self.assertRaises(PreventUpdate):
            show_delete_planning_confirmation([1], [
                {"type": "wrong-button", "session": "Planning Session 1"}
            ])
    
    @patch('dash_app.callback_context')
    def test_show_delete_confirmation_no_clicks(self, mock_ctx):
        """Test that no clicks prevents update."""
        mock_ctx.triggered_id = {"type": "planning-menu-button", "session": "Planning Session 1"}
        
        from dash.exceptions import PreventUpdate
        
        with self.assertRaises(PreventUpdate):
            show_delete_planning_confirmation([0], [
                {"type": "planning-menu-button", "session": "Planning Session 1"}
            ])


@unittest.skipUnless(DASH_AVAILABLE, "Dash not available - run in synth_experiments conda env")
class TestDataStructures(unittest.TestCase):
    """Test planning session data structures."""
    
    def test_planning_session_structure(self):
        """Test that planning sessions have correct structure."""
        session = {
            "name": "Planning Session 1",
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Verify required fields
        self.assertIn("name", session)
        self.assertIn("messages", session)
        self.assertIn("created_at", session)
        self.assertIn("updated_at", session)
        
        # Verify message structure
        for msg in session["messages"]:
            self.assertIn("role", msg)
            self.assertIn("content", msg)
            self.assertIn(msg["role"], ["user", "assistant"])


if __name__ == "__main__":
    if DASH_AVAILABLE:
        print("✓ Dash environment detected - running integration tests")
        unittest.main()
    else:
        print("✗ Dash not available - skipping integration tests")
        print("  Activate synth_experiments conda environment to run these tests")


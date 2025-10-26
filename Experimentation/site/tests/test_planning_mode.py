"""Tests for planning mode functionality.

This module contains UNIT TESTS that can run in any Python environment.
These tests mock external dependencies and don't require dash to be installed.

For INTEGRATION TESTS that import dash_app directly, see test_planning_integration.py
which requires the synth_experiments conda environment.

To run these unit tests (works anywhere):
    python -m pytest tests/test_planning_mode.py -v

To run all planning tests (requires conda env):
    conda activate synth_experiments
    python -m pytest tests/test_planning_*.py -v
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Test planning storage functions
class TestPlanningStorage(unittest.TestCase):
    """Test planning session storage functions."""
    
    @patch('storage.Path')
    def test_load_planning_sessions_empty(self, mock_path):
        """Test loading planning sessions when file doesn't exist."""
        from storage import load_planning_sessions
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with patch('storage._planning_store_path', return_value=mock_path_instance):
            sessions = load_planning_sessions("test_user")
            self.assertEqual(sessions, [])
    
    @patch('storage.Path')
    def test_save_and_load_planning_sessions(self, mock_path):
        """Test saving and loading planning sessions."""
        from storage import save_planning_sessions, load_planning_sessions
        
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        
        test_sessions = [
            {
                "name": "Planning Session 1",
                "messages": [
                    {"role": "user", "content": "How do I choose controls?"},
                    {"role": "assistant", "content": "Consider these factors..."},
                ],
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T01:00:00",
            }
        ]
        
        stored_data = {}
        
        def mock_write_text(content, encoding=None):
            stored_data['content'] = json.loads(content)
        
        def mock_read_text(encoding=None):
            return json.dumps(stored_data.get('content', {}))
        
        mock_path_instance.write_text = mock_write_text
        mock_path_instance.read_text = mock_read_text
        mock_path_instance.exists.return_value = True
        
        with patch('storage._planning_store_path', return_value=mock_path_instance):
            # Save sessions
            save_planning_sessions("test_user", test_sessions)
            
            # Load sessions
            loaded_sessions = load_planning_sessions("test_user")
            
            self.assertEqual(len(loaded_sessions), 1)
            self.assertEqual(loaded_sessions[0]["name"], "Planning Session 1")
            self.assertEqual(len(loaded_sessions[0]["messages"]), 2)


# Test planning validator
class TestPlanningValidator(unittest.TestCase):
    """Test planning validator service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        
    def test_validate_valid_question(self):
        """Test validating a valid planning question."""
        from services.planning_validator import PlanningValidator
        
        # Mock response
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": True,
            "reason": ""
        })
        self.mock_client.responses.parse.return_value = mock_response
        
        validator = PlanningValidator(self.mock_client)
        should_continue, reason = validator.validate_question(
            "What quasi-experimental method should I use for my study?"
        )
        
        self.assertTrue(should_continue)
        self.assertEqual(reason, "")
    
    def test_validate_invalid_question(self):
        """Test validating an invalid (off-topic) question."""
        from services.planning_validator import PlanningValidator
        
        # Mock response
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "should_continue": False,
            "reason": "I can only help with questions about planning and designing quasi-experimental studies."
        })
        self.mock_client.responses.parse.return_value = mock_response
        
        validator = PlanningValidator(self.mock_client)
        should_continue, reason = validator.validate_question(
            "What's the weather like today?"
        )
        
        self.assertFalse(should_continue)
        self.assertIn("planning and designing", reason)
    
    def test_validate_empty_question(self):
        """Test validating an empty question."""
        from services.planning_validator import PlanningValidator
        
        validator = PlanningValidator(self.mock_client)
        should_continue, reason = validator.validate_question("")
        
        self.assertFalse(should_continue)
        self.assertEqual(reason, "Question cannot be empty.")


# Test planning chat service
class TestPlanningChatService(unittest.TestCase):
    """Test planning chat service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
    
    def test_chat_with_no_history(self):
        """Test chat with no previous messages."""
        from services.planning_chat import PlanningChatService
        
        # Mock response
        mock_response = Mock()
        mock_response.output_text = "Here's my advice on selecting controls..."
        self.mock_client.responses.create.return_value = mock_response
        
        service = PlanningChatService(self.mock_client)
        response = service.chat([], "How do I select control groups?")
        
        self.assertIn("advice", response)
        self.mock_client.responses.create.assert_called_once()
    
    def test_chat_with_history(self):
        """Test chat with conversation history."""
        from services.planning_chat import PlanningChatService
        
        # Mock response
        mock_response = Mock()
        mock_response.output_text = "Based on our previous discussion..."
        self.mock_client.responses.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "What is synthetic control?"},
            {"role": "assistant", "content": "Synthetic control is a method..."},
        ]
        
        service = PlanningChatService(self.mock_client)
        response = service.chat(messages, "Can you explain more about when to use it?")
        
        self.assertIn("previous discussion", response)
        
        # Verify conversation history was included
        call_args = self.mock_client.responses.create.call_args
        conversation = call_args[1]['input']
        
        # Should have system prompt + 2 history messages + 1 new question = 4 total
        self.assertEqual(len(conversation), 4)
    
    def test_chat_error_handling(self):
        """Test chat service error handling."""
        from services.planning_chat import PlanningChatService
        
        # Mock error
        self.mock_client.responses.create.side_effect = Exception("API Error")
        
        service = PlanningChatService(self.mock_client)
        response = service.chat([], "Test question")
        
        self.assertIn("error", response.lower())


# Test helper functions
class TestPlanningHelpers(unittest.TestCase):
    """Test planning helper functions."""
    
    def test_generate_planning_session_name(self):
        """Test planning session name generation."""
        # Test the naming pattern
        name1 = f"Planning Session {1}"
        name2 = f"Planning Session {5}"
        
        self.assertEqual(name1, "Planning Session 1")
        self.assertEqual(name2, "Planning Session 5")
    
    def test_session_structure(self):
        """Test planning session data structure."""
        # Test that a planning session has the correct structure
        session = {
            "name": "Planning Session 1",
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Verify structure
        self.assertIn("name", session)
        self.assertIn("messages", session)
        self.assertIn("created_at", session)
        self.assertIn("updated_at", session)
        
        # Verify messages format
        self.assertEqual(len(session["messages"]), 2)
        self.assertEqual(session["messages"][0]["role"], "user")
        self.assertEqual(session["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()


"""Planning chat service for interactive conversations about experiment planning and design."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PlanningChatService:
    """Service for chatting about experiment planning and quasi-experimental design."""
    
    def __init__(self, client):
        """Initialize the planning chat service.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        user_question: str
    ) -> str:
        """Generate a response to user question about experiment planning.
        
        Args:
            messages: List of previous chat messages (for history)
            user_question: The user's current question
            
        Returns:
            Assistant's response as a string
        """
        # Build system prompt for experiment planning
        system_prompt = (
            "You are an expert consultant specializing in quasi-experimental design and causal inference. "
            "Your role is to help users plan and design their experiments by providing thoughtful, "
            "methodologically sound advice.\n\n"
            
            "Your expertise includes:\n"
            "- Quasi-experimental methods (Difference-in-Differences, Synthetic Control, Interrupted Time Series, etc.)\n"
            "- Experimental design principles and best practices\n"
            "- Causal inference frameworks and assumptions\n"
            "- Control group selection strategies\n"
            "- Sample size considerations and statistical power\n"
            "- Validity concerns (internal, external, construct, statistical)\n"
            "- Data requirements for different methods\n"
            "- Common pitfalls and how to avoid them\n\n"
            
            "When providing advice:\n"
            "- Ask clarifying questions if the user's scenario is not fully specified\n"
            "- Explain trade-offs between different methodological choices\n"
            "- Be practical and acknowledge real-world constraints\n"
            "- Reference key assumptions that must be satisfied\n"
            "- Suggest appropriate methods based on the user's context\n"
            "- Be clear but not overly technical unless asked\n"
            "- Encourage good experimental hygiene (pre-registration, robustness checks, etc.)\n\n"
            
            "Remember: You're helping with planning and design, not analyzing actual data. "
            "Focus on methodology, approach selection, and design considerations."
        )
        
        # Build message history
        conversation = [{"role": "system", "content": system_prompt}]
        
        # Add previous messages (conversation history)
        for msg in messages:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current question
        conversation.append({
            "role": "user",
            "content": user_question
        })
        
        try:
            logger.info(f"Sending planning chat request with {len(messages)} previous messages")
            
            response = self.client.responses.create(
                model="gpt-4o-mini",
                input=conversation
            )
            
            assistant_response = response.output_text
            logger.info(f"Received planning response: {len(assistant_response)} characters")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating planning chat response: {e}")
            return f"⚠️ Sorry, I encountered an error while processing your question: {str(e)}"


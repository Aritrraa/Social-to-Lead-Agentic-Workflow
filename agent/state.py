"""
agent/state.py
State management for the AutoStream Conversational AI Agent.
Defines the AgentState TypedDict used across all LangGraph nodes.
"""

from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import BaseMessage


# Intent categories
IntentType = Literal[
    "greeting",
    "product_inquiry",
    "pricing_inquiry",
    "high_intent_lead",
    "support_query",
    "off_topic",
    "unknown"
]

# Lead collection stages
LeadStage = Literal[
    "not_started",
    "awaiting_name",
    "awaiting_email",
    "awaiting_platform",
    "complete"
]


class AgentState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.
    Persisted across the entire conversation.
    """
    # Conversation history (list of HumanMessage / AIMessage objects)
    messages: List[BaseMessage]

    # Detected intent of the most recent user message
    current_intent: Optional[IntentType]

    # Running summary of conversation context (for long conversations)
    conversation_summary: Optional[str]

    # Lead collection state
    lead_stage: LeadStage
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Whether the lead capture tool has been triggered
    lead_captured: bool

    # The last retrieved RAG context (for debugging / transparency)
    last_rag_context: Optional[str]

    # Turn counter for conversation management
    turn_count: int


def get_initial_state() -> AgentState:
    """Return a fresh initial state for a new conversation."""
    return {
        "messages": [],
        "current_intent": None,
        "conversation_summary": None,
        "lead_stage": "not_started",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "last_rag_context": None,
        "turn_count": 0,
    }

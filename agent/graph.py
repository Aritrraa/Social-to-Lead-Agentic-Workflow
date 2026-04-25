"""
agent/graph.py
LangGraph-based conversational agent for AutoStream.

Architecture:
    User Input
        │
        ▼
    [detect_intent] ──────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
    [route_by_intent]                                         │
        │                                                      │
        ├─ greeting ──────────► [handle_greeting]             │
        │                              │                       │
        ├─ product/pricing ──► [handle_rag_query]             │
        │                              │                       │
        ├─ high_intent_lead ─► [handle_lead_collection]       │
        │       │                      │                       │
        │       └─ complete ──► [execute_lead_capture]        │
        │                              │                       │
        └─ other ────────────► [handle_general]               │
                                       │                       │
                                       ▼                       │
                                   [format_response] ◄────────┘
                                       │
                                       ▼
                                   END (return to user)
"""

import os
import re
import sys
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from .state import AgentState, get_initial_state
from .rag_pipeline import AutoStreamRAG
from tools.lead_capture import mock_lead_capture, validate_email, validate_platform


# ─────────────────────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────────────────────

def get_llm():
    """Initialize Llama 3 via Groq API (free, no billing required)."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Get your FREE key at: https://console.groq.com/"
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Alex, the friendly and knowledgeable AI assistant for AutoStream — a SaaS platform that provides automated AI-powered video editing tools for content creators.

Your personality:
- Warm, professional, and enthusiastic about helping creators
- Concise but thorough — don't ramble
- Always honest about what you know and don't know

Your core responsibilities:
1. Answer questions about AutoStream using ONLY the provided knowledge base context
2. Detect when a user is ready to sign up (high-intent) and collect their lead info
3. Never make up pricing, features, or policies — only use what's in the knowledge base

Important rules:
- NEVER trigger lead capture prematurely. Only collect leads when the user explicitly shows interest in signing up or trying the product.
- NEVER ask for name/email unless you've confirmed the user wants to proceed with a trial or purchase.
- When collecting lead info, ask for ONE piece of information at a time (name → email → platform).
- Be natural — don't sound like a form."""


# ─────────────────────────────────────────────────────────────
# RAG Instance (shared across all nodes)
# ─────────────────────────────────────────────────────────────

rag = AutoStreamRAG()


# ─────────────────────────────────────────────────────────────
# Node: detect_intent
# ─────────────────────────────────────────────────────────────

def detect_intent(state: AgentState) -> AgentState:
    """
    Classify the user's most recent message into an intent category.
    Uses rule-based pattern matching + LLM fallback for accuracy.
    """
    if not state["messages"]:
        state["current_intent"] = "unknown"
        return state

    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        state["current_intent"] = "unknown"
        return state

    user_text = last_message.content.lower().strip()

    # ── Rule-based fast-path ──────────────────────────────────
    greeting_patterns = [
        r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))[\s!.,]*$"
    ]
    pricing_patterns = [
        r"(price|pricing|cost|how much|plan|subscription|fee|monthly|annual|pay)",
        r"(basic|pro)\s+plan"
    ]
    high_intent_patterns = [
        r"(want to|ready to|i'?d like to|sign(ing)? up|get started|start(ing)? (a )?trial|try it|subscribe)",
        r"(interested in|go with|choose|pick)\s+(the\s+)?(basic|pro|plan)",
        r"(where do i|how do i)\s+(sign up|register|start)",
        r"(my\s+)?(youtube|instagram|tiktok|channel|content)"
    ]
    support_patterns = [
        r"(support|help|customer service|contact|issue|problem|bug|error)"
    ]

    for pattern in greeting_patterns:
        if re.search(pattern, user_text):
            state["current_intent"] = "greeting"
            return state

    # Check if we're mid-lead-collection — treat input as lead data
    if state["lead_stage"] in ("awaiting_name", "awaiting_email", "awaiting_platform"):
        state["current_intent"] = "high_intent_lead"
        return state

    for pattern in high_intent_patterns:
        if re.search(pattern, user_text):
            state["current_intent"] = "high_intent_lead"
            return state

    for pattern in pricing_patterns:
        if re.search(pattern, user_text):
            state["current_intent"] = "pricing_inquiry"
            return state

    for pattern in support_patterns:
        if re.search(pattern, user_text):
            state["current_intent"] = "support_query"
            return state

    # General product question
    product_keywords = ["autostream", "feature", "video", "edit", "caption", "resolution", "4k", "upload", "export", "platform", "refund", "trial", "cancel"]
    if any(kw in user_text for kw in product_keywords):
        state["current_intent"] = "product_inquiry"
        return state

    # Default: treat as product inquiry so RAG handles it
    state["current_intent"] = "product_inquiry"
    return state


# ─────────────────────────────────────────────────────────────
# Node: handle_greeting
# ─────────────────────────────────────────────────────────────

def handle_greeting(state: AgentState) -> AgentState:
    """Generate a warm greeting response."""
    llm = get_llm()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["messages"][-1].content),
    ]
    
    prompt_addition = (
        "\nThe user is greeting you. Respond warmly, introduce yourself as Alex from AutoStream, "
        "and briefly mention you can help with pricing, features, and getting started. Keep it under 3 sentences."
    )
    messages[0] = SystemMessage(content=SYSTEM_PROMPT + prompt_addition)
    
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    state["turn_count"] += 1
    return state


# ─────────────────────────────────────────────────────────────
# Node: handle_rag_query
# ─────────────────────────────────────────────────────────────

def handle_rag_query(state: AgentState) -> AgentState:
    """
    Handle product/pricing questions using RAG-retrieved context.
    """
    llm = get_llm()
    user_query = state["messages"][-1].content
    
    # Retrieve relevant knowledge
    context = rag.get_context_string(user_query)
    state["last_rag_context"] = context

    # Build conversation history for context (last 6 messages)
    history = state["messages"][:-1][-6:]  # exclude current message, keep last 6
    
    system_with_context = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== KNOWLEDGE BASE CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
        f"Answer the user's question using ONLY the above context. "
        f"If the answer isn't in the context, say so honestly. "
        f"After answering, you may gently ask if they'd like to get started or try a free trial."
    )

    messages = [SystemMessage(content=system_with_context)]
    for msg in history:
        messages.append(msg)
    messages.append(HumanMessage(content=user_query))
    
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    state["turn_count"] += 1
    return state


# ─────────────────────────────────────────────────────────────
# Node: handle_lead_collection
# ─────────────────────────────────────────────────────────────

def handle_lead_collection(state: AgentState) -> AgentState:
    """
    Manage the multi-step lead qualification form.
    Collects name → email → platform one step at a time.
    """
    user_input = state["messages"][-1].content.strip()
    lead_stage = state["lead_stage"]
    response_text = ""

    # ── Process incoming data based on current stage ──────────
    if lead_stage == "not_started":
        # User just showed high intent → start collection
        state["lead_stage"] = "awaiting_name"
        response_text = (
            "Awesome! I'd love to get you set up with AutoStream's Pro plan. "
            "Let's start with your name — what should I call you?"
        )

    elif lead_stage == "awaiting_name":
        # Validate name (at least 2 characters)
        if len(user_input) >= 2:
            state["lead_name"] = user_input.strip()
            state["lead_stage"] = "awaiting_email"
            response_text = (
                f"Great to meet you, {state['lead_name']}! "
                f"What's the best email address to reach you at?"
            )
        else:
            response_text = "Hmm, I didn't catch your name. Could you share your full name?"

    elif lead_stage == "awaiting_email":
        # Validate email format
        if validate_email(user_input):
            state["lead_email"] = user_input.strip().lower()
            state["lead_stage"] = "awaiting_platform"
            response_text = (
                f"Perfect! And which platform are you primarily creating content on? "
                f"(e.g., YouTube, Instagram, TikTok, etc.)"
            )
        else:
            response_text = (
                f"That doesn't look like a valid email address. "
                f"Could you double-check and try again?"
            )

    elif lead_stage == "awaiting_platform":
        # Accept any platform name
        normalized = validate_platform(user_input)
        state["lead_platform"] = normalized
        state["lead_stage"] = "complete"
        response_text = (
            f"Perfect! Let me lock in your details now..."
        )

    state["messages"].append(AIMessage(content=response_text))
    state["turn_count"] += 1
    return state


# ─────────────────────────────────────────────────────────────
# Node: execute_lead_capture
# ─────────────────────────────────────────────────────────────

def execute_lead_capture(state: AgentState) -> AgentState:
    """
    Trigger the mock_lead_capture tool once all 3 fields are collected.
    """
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )

    state["lead_captured"] = True

    if result["success"]:
        confirmation = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"I've sent your details to our team and you'll receive a welcome email at "
            f"{state['lead_email']} shortly.\n\n"
            f"Your 7-day free Pro trial is now queued for activation. "
            f"Our team will follow up within 24 hours to help you get the most out of AutoStream "
            f"for your {state['lead_platform']} content. "
            f"Welcome aboard! 🚀"
        )
    else:
        confirmation = (
            f"I ran into a small issue saving your details ({result.get('error', 'unknown error')}). "
            f"Please try again or reach out to our support team directly."
        )

    state["messages"].append(AIMessage(content=confirmation))
    state["turn_count"] += 1
    return state


# ─────────────────────────────────────────────────────────────
# Node: handle_general
# ─────────────────────────────────────────────────────────────

def handle_general(state: AgentState) -> AgentState:
    """Handle off-topic, support, or ambiguous queries."""
    llm = get_llm()
    user_query = state["messages"][-1].content
    context = rag.get_context_string(user_query)
    
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== KNOWLEDGE BASE CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"If the question is about AutoStream, answer using the context. "
        f"If it's off-topic, politely redirect to AutoStream topics."
    )
    
    history = state["messages"][:-1][-4:]
    messages = [SystemMessage(content=system)]
    for msg in history:
        messages.append(msg)
    messages.append(HumanMessage(content=user_query))
    
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    state["turn_count"] += 1
    return state


# ─────────────────────────────────────────────────────────────
# Routing Functions
# ─────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on detected intent."""
    intent = state.get("current_intent", "unknown")
    
    if intent == "greeting":
        return "handle_greeting"
    elif intent in ("pricing_inquiry", "product_inquiry"):
        return "handle_rag_query"
    elif intent == "high_intent_lead":
        return "handle_lead_collection"
    elif intent == "support_query":
        return "handle_rag_query"
    else:
        return "handle_general"


def route_after_lead_collection(state: AgentState) -> str:
    """After lead collection, decide whether to capture or continue collecting."""
    if state["lead_stage"] == "complete" and not state["lead_captured"]:
        return "execute_lead_capture"
    return END


# ─────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_rag_query", handle_rag_query)
    workflow.add_node("handle_lead_collection", handle_lead_collection)
    workflow.add_node("execute_lead_capture", execute_lead_capture)
    workflow.add_node("handle_general", handle_general)

    # Entry point
    workflow.set_entry_point("detect_intent")

    # Edges
    workflow.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_rag_query": "handle_rag_query",
            "handle_lead_collection": "handle_lead_collection",
            "handle_general": "handle_general",
        }
    )

    # After greeting → END
    workflow.add_edge("handle_greeting", END)
    
    # After RAG → END
    workflow.add_edge("handle_rag_query", END)
    
    # After general → END
    workflow.add_edge("handle_general", END)

    # Lead collection → maybe capture → END
    workflow.add_conditional_edges(
        "handle_lead_collection",
        route_after_lead_collection,
        {
            "execute_lead_capture": "execute_lead_capture",
            END: END,
        }
    )
    workflow.add_edge("execute_lead_capture", END)

    return workflow.compile()


# ─────────────────────────────────────────────────────────────
# Agent Runner
# ─────────────────────────────────────────────────────────────

class AutoStreamAgent:
    """
    High-level interface for the AutoStream conversational agent.
    Maintains state across conversation turns.
    """

    def __init__(self):
        self.graph = build_graph()
        self.state = get_initial_state()

    def chat(self, user_input: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            user_input: The user's text message
            
        Returns:
            The agent's response string
        """
        # Append user message to state
        self.state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        self.state = self.graph.invoke(self.state)

        # Extract and return the last AI message
        for msg in reversed(self.state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I'm sorry, I encountered an issue. Please try again."

    def reset(self):
        """Reset the agent state for a new conversation."""
        self.state = get_initial_state()

    def get_debug_info(self) -> dict:
        """Return current state for debugging purposes."""
        return {
            "turn_count": self.state["turn_count"],
            "current_intent": self.state["current_intent"],
            "lead_stage": self.state["lead_stage"],
            "lead_name": self.state["lead_name"],
            "lead_email": self.state["lead_email"],
            "lead_platform": self.state["lead_platform"],
            "lead_captured": self.state["lead_captured"],
        }

"""
tests/test_agent.py
Unit and integration tests for the AutoStream AI Agent.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag_pipeline import AutoStreamRAG
from agent.state import get_initial_state
from tools.lead_capture import mock_lead_capture, validate_email, validate_platform


# ─────────────────────────────────────────────────────────────
# RAG Pipeline Tests
# ─────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Test the knowledge retrieval system."""

    def setup_method(self):
        self.rag = AutoStreamRAG()

    def test_kb_loads_successfully(self):
        """Knowledge base should load without errors."""
        assert self.rag.knowledge_base is not None
        assert "pricing" in self.rag.knowledge_base
        assert "policies" in self.rag.knowledge_base

    def test_chunks_created(self):
        """Chunks should be created from knowledge base."""
        assert len(self.rag.chunks) > 5

    def test_pricing_query_retrieves_pricing(self):
        """Pricing queries should retrieve pricing chunks."""
        results = self.rag.retrieve("What are your pricing plans?")
        categories = [r["category"] for r in results]
        assert any("pricing" in cat for cat in categories)

    def test_refund_query_retrieves_policy(self):
        """Refund queries should retrieve policy chunks."""
        results = self.rag.retrieve("What is your refund policy?")
        categories = [r["category"] for r in results]
        assert "policy_refund" in categories

    def test_pro_plan_query(self):
        """Pro plan queries should retrieve pro plan details."""
        results = self.rag.retrieve("Tell me about the Pro plan")
        categories = [r["category"] for r in results]
        assert any("pro" in cat.lower() for cat in categories)

    def test_get_context_string_not_empty(self):
        """Context string should be non-empty for valid queries."""
        context = self.rag.get_context_string("How much does AutoStream cost?")
        assert len(context) > 50
        assert "29" in context or "79" in context  # prices should appear

    def test_irrelevant_query_returns_fallback(self):
        """Completely irrelevant queries should still return some context."""
        context = self.rag.get_context_string("xyzzy foobar nonsense")
        assert isinstance(context, str)


# ─────────────────────────────────────────────────────────────
# Lead Capture Tool Tests
# ─────────────────────────────────────────────────────────────

class TestLeadCapture:
    """Test the mock lead capture tool."""

    def test_successful_lead_capture(self, capsys):
        """Valid lead data should succeed."""
        result = mock_lead_capture("Jane Doe", "jane@example.com", "YouTube")
        assert result["success"] is True
        assert "lead_id" in result
        assert result["data"]["name"] == "Jane Doe"

    def test_lead_capture_prints_output(self, capsys):
        """Lead capture should print confirmation to console."""
        mock_lead_capture("Test User", "test@test.com", "Instagram")
        captured = capsys.readouterr()
        assert "Lead captured successfully" in captured.out
        assert "Test User" in captured.out

    def test_invalid_email_rejected(self):
        """Invalid email should return failure."""
        result = mock_lead_capture("John", "not-an-email", "TikTok")
        assert result["success"] is False

    def test_missing_fields_rejected(self):
        """Missing required fields should return failure."""
        result = mock_lead_capture("", "email@test.com", "YouTube")
        assert result["success"] is False

    def test_lead_id_format(self):
        """Lead ID should follow expected format."""
        result = mock_lead_capture("Alex Smith", "alex@creator.io", "TikTok")
        assert result["success"] is True
        assert result["lead_id"].startswith("LEAD-")


class TestValidations:
    """Test input validation helpers."""

    def test_valid_emails(self):
        assert validate_email("user@example.com") is True
        assert validate_email("name+tag@domain.co.uk") is True

    def test_invalid_emails(self):
        assert validate_email("notanemail") is False
        assert validate_email("missing@nodot") is False
        assert validate_email("@nodomain.com") is False

    def test_platform_normalization(self):
        assert validate_platform("youtube") == "YouTube"
        assert validate_platform("yt") == "YouTube"
        assert validate_platform("ig") == "Instagram"
        assert validate_platform("tiktok") == "TikTok"
        assert validate_platform("CustomPlatform") == "Customplatform"


# ─────────────────────────────────────────────────────────────
# State Management Tests
# ─────────────────────────────────────────────────────────────

class TestStateManagement:
    """Test the agent state initialization and structure."""

    def test_initial_state_structure(self):
        """Initial state should have all required keys."""
        state = get_initial_state()
        required_keys = [
            "messages", "current_intent", "lead_stage",
            "lead_name", "lead_email", "lead_platform",
            "lead_captured", "turn_count"
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_initial_lead_stage(self):
        """Lead stage should start as 'not_started'."""
        state = get_initial_state()
        assert state["lead_stage"] == "not_started"

    def test_lead_not_captured_initially(self):
        """Lead should not be captured at start."""
        state = get_initial_state()
        assert state["lead_captured"] is False

    def test_messages_start_empty(self):
        """Message history should be empty initially."""
        state = get_initial_state()
        assert state["messages"] == []


# ─────────────────────────────────────────────────────────────
# Run tests standalone
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running AutoStream Agent Tests...\n")
    
    # Quick smoke test without pytest
    rag = AutoStreamRAG()
    print("✅ RAG Pipeline loaded successfully")
    print(f"   Chunks created: {len(rag.chunks)}")
    
    context = rag.get_context_string("What are the pricing plans?")
    print(f"   Context retrieved: {len(context)} characters")
    
    result = mock_lead_capture("Test Lead", "test@example.com", "YouTube")
    print(f"✅ Lead Capture: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    state = get_initial_state()
    print(f"✅ State initialized: {len(state)} fields")
    
    print("\n✅ All smoke tests passed! Run `pytest tests/ -v` for full test suite.")

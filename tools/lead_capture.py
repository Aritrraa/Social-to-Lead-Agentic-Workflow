"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
"""

import json
import os
from datetime import datetime
from typing import Optional


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture qualified leads.
    
    Args:
        name: Full name of the lead
        email: Email address of the lead  
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)
    
    Returns:
        dict: Response confirming lead capture with lead_id
    """
    # Validate inputs
    if not name or not email or not platform:
        return {
            "success": False,
            "error": "Missing required fields: name, email, platform"
        }
    
    # Basic email format check
    if "@" not in email or "." not in email.split("@")[-1]:
        return {
            "success": False,
            "error": f"Invalid email format: {email}"
        }

    # Generate a mock lead ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    lead_id = f"LEAD-{timestamp}-{name[:3].upper()}"

    # Simulate saving to a CRM / database
    lead_data = {
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat(),
        "source": "AI Agent - Social Conversation",
        "status": "new",
        "interested_plan": "Pro"
    }

    # Save to local file (simulating DB write)
    _save_lead_to_file(lead_data)

    # Console output as required by assignment
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"   Lead ID: {lead_id}")
    print(f"   Timestamp: {lead_data['captured_at']}")
    print(f"{'='*50}\n")

    return {
        "success": True,
        "lead_id": lead_id,
        "message": f"Lead successfully captured for {name}",
        "data": lead_data
    }


def _save_lead_to_file(lead_data: dict) -> None:
    """Save lead to a local JSON file (mock database)."""
    leads_file = os.path.join(os.path.dirname(__file__), "..", "data", "leads.json")
    os.makedirs(os.path.dirname(leads_file), exist_ok=True)

    # Load existing leads
    leads = []
    if os.path.exists(leads_file):
        try:
            with open(leads_file, "r") as f:
                leads = json.load(f)
        except (json.JSONDecodeError, IOError):
            leads = []

    # Append new lead
    leads.append(lead_data)

    # Save back
    with open(leads_file, "w") as f:
        json.dump(leads, f, indent=2)


def validate_email(email: str) -> bool:
    """Basic email validation."""
    return "@" in email and "." in email.split("@")[-1]


def validate_platform(platform: str) -> Optional[str]:
    """
    Normalize and validate creator platform input.
    Returns normalized name or None if unrecognized.
    """
    known_platforms = {
        "youtube": "YouTube",
        "yt": "YouTube",
        "instagram": "Instagram",
        "ig": "Instagram",
        "insta": "Instagram",
        "tiktok": "TikTok",
        "tt": "TikTok",
        "twitter": "Twitter/X",
        "x": "Twitter/X",
        "facebook": "Facebook",
        "fb": "Facebook",
        "twitch": "Twitch",
        "linkedin": "LinkedIn",
    }
    normalized = platform.strip().lower()
    return known_platforms.get(normalized, platform.title())  # Return as-is if not in dict

"""
main.py
Entry point for the AutoStream Conversational AI Agent.

Usage:
    python main.py           # Interactive CLI chat
    python main.py --demo    # Run a scripted demo conversation
"""

import os
import sys
import argparse
from dotenv import load_dotenv  # ADD THIS
load_dotenv()          


def check_env():
    """Verify required environment variables are set."""
    if not os.environ.get("GROQ_API_KEY"):
        print("\n❌ ERROR: GROQ_API_KEY environment variable not set.")
        print("   On Windows (PowerShell), run:")
        print('   $env:GROQ_API_KEY = "your-key-here"')
        print("   On Mac/Linux, run:")
        print('   export GROQ_API_KEY="your-key-here"')
        print("   Get your free key at: https://console.groq.com/\n")
        sys.exit(1)


def run_interactive():
    """Run the agent in interactive CLI mode."""
    from agent.graph import AutoStreamAgent

    print("\n" + "="*60)
    print("  🎬 AutoStream AI Assistant (Powered by Llama 3.1 via Groq (Free))")
    print("="*60)
    print("  Type your message and press Enter to chat.")
    print("  Commands: 'quit' or 'exit' to stop | 'reset' to restart | 'debug' for state")
    print("="*60 + "\n")

    agent = AutoStreamAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAlex: Thanks for chatting! Have a great day! 🎬\n")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("\n[Conversation reset. Starting fresh.]\n")
            continue

        if user_input.lower() == "debug":
            info = agent.get_debug_info()
            print("\n[DEBUG STATE]")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print()
            continue

        print("\nAlex: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()


def run_demo():
    """Run a scripted demo showing all agent capabilities."""
    from agent.graph import AutoStreamAgent

    demo_script = [
        ("Hi there!", "Greeting → Agent introduces itself"),
        ("Tell me about your pricing plans.", "RAG → Pricing retrieval"),
        ("What's included in the Pro plan exactly?", "RAG → Pro plan details"),
        ("Is there a refund policy?", "RAG → Policy retrieval"),
        ("That sounds great! I want to try the Pro plan for my YouTube channel.", "High Intent → Lead collection starts"),
        ("Sarah Johnson", "Lead → Name collected"),
        ("sarah@creatorstudio.com", "Lead → Email collected"),
        ("YouTube", "Lead → Platform collected → Lead captured!"),
    ]

    print("\n" + "="*65)
    print("  🎬 AutoStream Agent — DEMO MODE")
    print("="*65)
    print("  This demo simulates a complete sales conversation.")
    print("="*65 + "\n")

    agent = AutoStreamAgent()

    for i, (user_message, description) in enumerate(demo_script, 1):
        print(f"─── Turn {i}: [{description}] ───")
        print(f"You: {user_message}")
        print()

        response = agent.chat(user_message)
        print(f"Alex: {response}")
        print()

        debug = agent.get_debug_info()
        print(f"  [Intent: {debug['current_intent']} | Lead Stage: {debug['lead_stage']}]")
        print()

        if debug["lead_captured"]:
            print("="*65)
            print("  ✅ DEMO COMPLETE — Lead successfully captured!")
            print("="*65)
            break


def main():
    check_env()

    parser = argparse.ArgumentParser(description="AutoStream AI Agent")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a scripted demo conversation"
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_interactive()


if __name__ == "__main__":
    main()

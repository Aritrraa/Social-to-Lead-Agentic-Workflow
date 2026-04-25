# 🚀 AutoStream Social-to-Lead Agentic Workflow

AutoStream is a complete, intelligent Conversational AI Agent built to capture high-intent leads automatically. Powered by **LangGraph**, **Llama 3.1 (via Groq)**, and **FastAPI**, this agent can dynamically route user intents, answer product questions using a built-in RAG (Retrieval-Augmented Generation) knowledge base, and smoothly collect lead information.

It includes a fully featured backend API and a **premium, sleek Dark Mode UI** built with Vanilla HTML/CSS/JS.

![AutoStream UI Preview](https://img.shields.io/badge/UI-Sleek_Dark_Mode-6366F1?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Flow-blue?style=for-the-badge)

---

## ✨ Features

- 🧠 **Agentic Routing:** Uses LangGraph's `StateGraph` to intelligently route user queries based on intent (Greeting, Pricing, Support, High-Intent Lead).
- 📚 **RAG Integration:** Automatically fetches facts from a local knowledge base (`autostream_kb.json`) to accurately answer questions without hallucinating.
- 🎯 **Multi-step Lead Capture:** Detects when a user wants to buy or sign up, and seamlessly transitions into a lead-capture state machine (collects Name → Email → Platform).
- ⚡ **Lightning Fast:** Uses `ChatGroq` with `llama-3.1-8b-instant` for ultra-fast, completely free LLM inference.
- 🎨 **Premium UI:** A stunning glassmorphic chat interface with an active "Agent State" debug panel.
- 🚀 **1-Click Deploy:** Pre-configured with a `Procfile` and `requirements.txt` for immediate deployment to platforms like Render or Railway.

---

## 📂 Project Structure

```
autostream-agent/
├── agent/
│   ├── graph.py          # The LangGraph workflow & Nodes
│   ├── state.py          # Agent memory & state definitions
│   └── rag_pipeline.py   # RAG logic & context retrieval
├── knowledge_base/
│   └── autostream_kb.json # The factual knowledge the AI uses
├── static/
│   ├── index.html        # Frontend structure
│   ├── styles.css        # Premium dark mode styling
│   └── app.js            # Frontend chat and API logic
├── tools/
│   └── lead_capture.py   # Simulated CRM connection tools
├── api.py                # FastAPI web server and endpoints
├── main.py               # CLI version of the agent
├── Procfile              # Deployment configuration
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variables template
```

---

## 🛠️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aritrraa/Social-to-Lead-Agentic-Workflow.git
cd Social-to-Lead-Agentic-Workflow
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Copy the example environment file and add your Groq API key (you can get one for free at [console.groq.com](https://console.groq.com/)):
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## 💻 Running the Application

### Option A: The Web Interface (Recommended)
To run the full visual application with the beautiful chat interface:
```bash
uvicorn api:app --reload
```
Then, open your browser and go to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### Option B: The Command Line Interface
If you prefer testing the agent inside the terminal:
```bash
python main.py
```
To run an automated test script simulating a full user conversation:
```bash
python main.py --demo
```

---

## 🚀 Deployment (Render.com)

This project is completely ready for cloud deployment. 

1. Go to [Render.com](https://render.com/) and connect your GitHub account.
2. Click **New +** > **Web Service**.
3. Select this repository.
4. Render will auto-detect the configuration. Just ensure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** *(Leave empty, Render uses the Procfile automatically)*
5. Add your `GROQ_API_KEY` under the Environment Variables section.
6. Click **Deploy!**

---

## 🤖 How the Agent Works

The agent doesn't just blindly query an LLM. It acts as a controlled state machine:

1. **Intent Detection:** Every message passes through an intent router. 
2. **Dynamic Routing:** 
   - If you say "Hi", it routes to a lightweight greeting handler.
   - If you ask "How much?", it routes to the **RAG node** to fetch pricing data.
   - If you say "I want to sign up", it routes to the **Lead Collection node**, temporarily ignoring the LLM and hard-coding logic to collect your Name, Email, and Platform step-by-step.
3. **State Memory:** The entire chat history and the user's lead profile are stored in an `AgentState` dictionary, allowing the AI to "remember" context perfectly.

---
*Built with LangChain, LangGraph, Groq, and FastAPI.*

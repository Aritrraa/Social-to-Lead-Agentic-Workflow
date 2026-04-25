import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from agent.graph import AutoStreamAgent

app = FastAPI(title="AutoStream API")

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a global instance of the agent
# Note: In a production app, you might want session-based agents
# to handle multiple concurrent users, but for a simple frontend
# this is a good start.
agent = AutoStreamAgent()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    debug_info: dict

@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        return {"response": "Please enter a message.", "debug_info": {}}
    
    # Check for commands
    msg = request.message.strip().lower()
    if msg == "reset":
        agent.reset()
        return {"response": "Conversation reset. Starting fresh.", "debug_info": agent.get_debug_info()}
    
    response = agent.chat(request.message)
    debug_info = agent.get_debug_info()
    
    return {"response": response, "debug_info": debug_info}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

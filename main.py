from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="SteelBot Pro API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversations = {}

class ChatRequest(BaseModel):
    message: str
    personality: str = "professional"
    session_id: str = "default"

@app.get("/")
def home():
    return {"message": "SteelBot Pro API is running"}

@app.get("/ui")
def get_ui():
    return FileResponse("index.html")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if request.session_id not in conversations:
            conversations[request.session_id] = []

        history = conversations[request.session_id]

        system_prompt = f"You are SteelBot, a steel industry expert. Your personality is {request.personality}."

        history.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}] + history
        )

        ai_reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": ai_reply})

        return {
            "session_id": request.session_id,
            "you_said": request.message,
            "reply": ai_reply,
            "history_length": len(history)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"something went wrong: {str(e)}"
        )

@app.get("/history/{session_id}")
def get_history(session_id: str):
    history = conversations.get(session_id, [])
    return {
        "session_id": session_id,
        "total_messages": len(history),
        "history": history
    }

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
    return {"message": f"History cleared for session {session_id}"}

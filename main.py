from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import openai
import os

# Load API Key from environment variable (Vercel secure environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set the OPENAI_API_KEY in Vercel Environment Variables.")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

class Newsletter(BaseModel):
    subject: str
    sender: str
    content: str

class SummarizationRequest(BaseModel):
    newsletters: List[Newsletter]

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    if not request.newsletters:
        raise HTTPException(status_code=400, detail="No newsletters provided")

    # Build the text prompt for GPT-4
    prompt_text = "Summarize the following newsletters into a clean, readable digest:\n\n"
    for newsletter in request.newsletters:
        prompt_text += f"- **{newsletter.subject}** from *{newsletter.sender}*:\n{newsletter.content}\n\n"

    # Call GPT-4 for summarization
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Change to "gpt-3.5-turbo" for cheaper option
        messages=[
            {"role": "system", "content": "You are an expert newsletter curator."},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=700,
        temperature=0.5,
    )

    summary = response.choices[0].message.content.strip()
    return {"summary": summary}

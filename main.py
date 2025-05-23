from fastapi import FastAPI, HTTPException, Query, Request, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
import datetime as dt
import logging
import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from google_auth_oauthlib.flow import InstalledAppFlow
import json, re

# Load API Key from environment variable (Vercel secure environment)
load_dotenv(override=True)

GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_API_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

if not GMAIL_CLIENT_ID or not GMAIL_CLIENT_SECRET or not GMAIL_REFRESH_TOKEN:
    raise RuntimeError("Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in Vercel Environment Variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set the OPENAI_API_KEY in Vercel Environment Variables.")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = FastAPI()
log = logging.getLogger("uvicorn.error")  # works on Vercel’s standard logger

class Newsletter(BaseModel):
    subject: str
    sender: str
    content: str

class SummarizationRequest(BaseModel):
    newsletters: List[Newsletter]

# @app.post("/summarize")
# async def summarize(request: SummarizationRequest):
#     if not request.newsletters:
#         raise HTTPException(status_code=400, detail="No newsletters provided")

#     # Build the text prompt for GPT-4
#     prompt_text = "Summarize the following newsletters into a clean, readable digest:\n\n"
#     for newsletter in request.newsletters:
#         prompt_text += f"- **{newsletter.subject}** from *{newsletter.sender}*:\n{newsletter.content}\n\n"

#     # Call GPT-4 for summarization
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",  # Change to "gpt-3.5-turbo" for cheaper option
#         messages=[
#             {"role": "system", "content": "You are an expert newsletter curator."},
#             {"role": "user", "content": prompt_text},
#         ],
#         max_tokens=700,
#         temperature=0.5,
#     )

#     summary = response.choices[0].message.content.strip()
#     return {"summary": summary}

class SummarizationResponse(BaseModel):
    tldr: List[str]
    topics: List[Dict[str, Any]]

@app.post("/summarize", response_model=SummarizationResponse)
# async def summarize(req: SummarizationRequest):
#     if not req.newsletters:
#         raise HTTPException(400, "No newsletters provided")

#     #–– Build the prompt ––#
#     prompt = "You are an expert curator. " \
#              "Group the news below into THEMES or TOPICS (e.g. Markets, Tech & AI, Global). " \
#              "For each theme, list key stories as JSON:\n\n"
#     for n in req.newsletters:
#         prompt += f"- From {n.sender}: {n.subject}\n{n.content[:]}\n\n"

#     completion = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role":"system","content":
#              "Return STRICT JSON with keys tldr (array of 3-6 bullets) "
#              "and topics (array of objects {name, items:[{headline,source}]})"},
#             {"role":"user","content":prompt}
#         ],
#         temperature=0.4,
#         max_tokens=900
#     )

#     # Parse JSON safely
#     import json, re
#     json_text = re.search(r'\{.*\}', completion.choices[0].message.content, re.S).group(0)
#     data = json.loads(json_text)
#     return data

async def summarize(req: SummarizationRequest):
    if not req.newsletters:
        raise HTTPException(400, "No newsletters provided")

    # --- Prompt construction ---
    prompt = (
        "You are an expert newsletter curator. Your job is to read newsletters from multiple sources and:\n"
        "1. Write a TL;DR with 4–6 *concise*, single-sentence bullets summarizing key takeaways or the most important news stories. "
        "Each bullet should be ≤ 18 words and must *not* be copy-pasted from the detailed stories.\n"
        "\n"
        "2. Then, find the most important stories and insights from newsletters. Group and write them into relevant THEMES (e.g., Tech, Markets, Policy, etc). Please have no more than 5 themes. If more than 5 is needed, make the 5th theme 'Miscellaneous'.\n"
        "   - Each story in each theme should be a **detailed multi-sentence summary**, ~3 sentences.\n"
        "   - Each story must include context, outcome, significance, and optionally source attribution.\n"
        "   - No bullet points in this section; use full sentences.\n"
        "Output format (strict JSON):\n"
        "{\n"
        "  \"tldr\": [\"...\", \"...\"],\n"
        "  \"topics\": [\n"
        "    {\"name\": \"Theme Name\", \"items\": [\n"
        "      { \"headline\": \"Multi-sentence description.\", \"source\": \"Source Name\" },\n"
        "      ...\n"
        "    ]}\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- TL;DR bullets must be unique and paraphrased (not repeated from headlines).\n"
        "- Each theme's stories should contain details than the TL;DR bullets. Overlap with the TL;DR bullets is okay.\n"
        "- Headline + source only; no extra fields.\n"
        "- Never include explanations, markdown, or commentary outside the JSON block.\n\n"
        "Newsletters:\n\n"
    )

    for n in req.newsletters:
        prompt += f"\n---\nFrom: {n.sender} | Subject: {n.subject}\n{n.content[:]}\n"

    # --- LLM call ---
    response = client.chat.completions.create(
        model="gpt-4o",  # Or gpt-4.1-mini if needed
        messages=[
            {"role": "system", "content": "You are a newsletter curator. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=5000
    )

    # --- Parse the JSON safely ---
    try:
        json_text = re.search(r'\{.*\}', response.choices[0].message.content, re.S).group(0)
        data = json.loads(json_text)
    except Exception as e:
        raise HTTPException(500, f"Failed to parse JSON from GPT: {e}")

    return data


##### Approach #2: Scheduled Task in GPT
#Gmail API Auth Helpers
def get_gmail_access_token():
    """Fetches an access token using the refresh token."""
    data = {
        "client_id": GMAIL_CLIENT_ID,
        "client_secret": GMAIL_CLIENT_SECRET,
        "refresh_token": GMAIL_REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    response = requests.post(GMAIL_API_TOKEN_URL, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print("Status:", response.status_code)
        print("Text:", response.text)
        raise HTTPException(status_code=500, detail="Failed to get Gmail access token.")


def extract_email_body(payload):
    """Recursively extracts the plain text or HTML body from the payload."""
    if 'parts' in payload:
        for part in payload['parts']:
            body = extract_email_body(part)
            if body:
                return body
    elif payload.get("mimeType") in ["text/plain", "text/html"]:
        data = payload.get("body", {}).get("data")
        if data:
            import base64
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return None


@app.get("/grab-newsletters")
async def grab_newsletters(hours_back: int = Query(24, ge=1, le=168)):
    """
    Fetch newsletters from Gmail labeled 'newsletters' within the last `hours_back` hours.
    """
    access_token = get_gmail_access_token()
    http_headers = {"Authorization": f"Bearer {access_token}"}

    # Format time correctly for Gmail query
    since = (dt.datetime.utcnow() - dt.timedelta(hours=hours_back)).strftime("%Y/%m/%d")
    query = f"label:newsletter after:{since}"

    # Fetch message list
    gmail_api_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    response = requests.get(gmail_api_url, headers=http_headers, params={"q": query})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch newsletters from Gmail.")

    messages = response.json().get("messages", [])
    if not messages:
        return {"newsletters": []}

    newsletters = []

    for message in messages[:10]:  # Limit to 10 for safety
        message_id = message.get("id")
        msg_detail_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
        msg_response = requests.get(msg_detail_url, headers=http_headers)

        if msg_response.status_code != 200:
            continue  # Skip this one and continue with the rest

        msg_data = msg_response.json()
        #snippet = msg_data.get("snippet", "")
        payload = msg_data.get("payload", {})
        content = extract_email_body(payload) or "[No content found]"
        
        # Correct: Use a new variable so we don't overwrite http_headers
        email_headers = msg_data.get("payload", {}).get("headers", [])
        subject = next((h['value'] for h in email_headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in email_headers if h['name'] == 'From'), "Unknown Sender")

        newsletters.append({
            "subject": subject,
            "sender": sender,
            "content": content
        })

    return {"newsletters": newsletters}

@app.get("/healthz")
async def health():
    return {"status": "ok"}
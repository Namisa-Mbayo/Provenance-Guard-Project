import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from groq import Groq


load_dotenv()

app = Flask(__name__)

LOG_PATH = Path("audit_log.json")
MIN_TEXT_LENGTH = 50


# ── Groq client ───────────────────────────────────────────────────────────────

def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    return Groq(api_key=api_key)


# ── audit log helpers ─────────────────────────────────────────────────────────

def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []

    try:
        with LOG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_log(entries: list[dict]) -> None:
    with LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)


def append_log_entry(entry: dict) -> None:
    entries = load_log()
    entries.append(entry)
    save_log(entries)


# ── first detection signal: LLM attribution ───────────────────────────────────

def classify_with_llm(text: str) -> dict:
    """
    First detection signal.

    Returns:
        {
            "score": float from 0 to 1,
            "attribution": "likely_ai" | "likely_human" | "uncertain",
            "reason": str
        }

    Score meaning:
        0 = very human-like
        1 = very AI-like
    """
    prompt = f"""
You are an AI authorship attribution classifier for a creative writing platform.

Analyze the submitted text and estimate how likely it is to be AI-generated.

Return ONLY valid JSON with these exact fields:
- score: a number between 0 and 1, where 0 means strongly human-written and 1 means strongly AI-generated
- attribution: one of "likely_ai", "likely_human", or "uncertain"
- reason: one concise sentence explaining the strongest evidence

Be cautious. Do not claim certainty unless the evidence is strong.
A polished human-written text may look AI-like, and informal AI-generated text may look human-like.

Submitted text:
{text}
"""

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cautious AI attribution classifier. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )

        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)

        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))

        attribution = result.get("attribution", "uncertain")
        if attribution not in {"likely_ai", "likely_human", "uncertain"}:
            attribution = score_to_attribution(score)

        reason = result.get("reason", "No reason provided.")

        return {
            "score": score,
            "attribution": attribution,
            "reason": reason,
        }

    except Exception as error:
        return {
            "score": 0.5,
            "attribution": "uncertain",
            "reason": f"LLM signal failed or could not be parsed: {str(error)}",
        }


# ── scoring helpers ───────────────────────────────────────────────────────────

def score_to_attribution(score: float) -> str:
    """
    Temporary Milestone 3 mapping based only on the LLM signal.
    Milestone 4 will replace this with combined scoring.
    """
    if score >= 0.75:
        return "likely_ai"

    if score <= 0.24:
        return "likely_human"

    return "uncertain"


def make_placeholder_label(attribution: str) -> str:
    """
    Placeholder label for Milestone 3.
    Milestone 5 will replace this with final transparency label variants.
    """
    if attribution == "likely_ai":
        return "Placeholder: This content shows signs of AI generation."

    if attribution == "likely_human":
        return "Placeholder: This content shows signs of human authorship."

    return "Placeholder: Authorship signals are mixed or uncertain."


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
def submit():
    data = request.get_json(silent=True) or {}

    # Milestone says use "text". This also accepts "content" for compatibility
    # with the earlier planning.md draft.
    text = data.get("text") or data.get("content")
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({
            "error": "Missing required field: user_id"
        }), 400

    if not text or not isinstance(text, str):
        return jsonify({
            "error": "Missing required field: text"
        }), 400

    if len(text.strip()) < MIN_TEXT_LENGTH:
        return jsonify({
            "error": f"Text is too short for attribution analysis. Please submit at least {MIN_TEXT_LENGTH} characters."
        }), 400

    content_id = f"content_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    llm_result = classify_with_llm(text)
    llm_score = llm_result["score"]
    attribution = llm_result["attribution"]
    confidence = llm_score
    label = make_placeholder_label(attribution)

    response_body = {
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": label,
        "llm_score": llm_score,
        "llm_reason": llm_result["reason"],
        "status": "classified",
    }

    log_entry = {
        "content_id": content_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "text": text,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "llm_reason": llm_result["reason"],
        "status": "classified",
    }

    append_log_entry(log_entry)

    return jsonify(response_body), 200


@app.get("/log")
def get_log():
    entries = load_log()
    return jsonify({"entries": entries[-20:]})


if __name__ == "__main__":
    app.run(debug=True)
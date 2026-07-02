# Provenance Guard Planning

## Architecture Narrative

When a user submits a piece of text, it goes through the `POST /submit` endpoint as a request. First, the submission is checked by the rate limiter to prevent spam. If the user is sending too many requests, the API returns a message about the rate limit.

If the request passes rate limiting, the API validates the input by checking if the text is long enough to analyze. If the text is too short, the API returns an error message. If the input is valid, the text goes through two detection signals. The first signal is an LLM based attribution classifier using Groq. This signal evaluates the semantics of the writing, such as tone, coherence, and whether the text resembles AI generated writing. The second signal is a stylometric heuristic analyzer. This signal measures structural properties of the text, such as sentence length variance, type token ratio, and punctuation.

The system then combines the two signal scores into a AI likelihood score. This score is used to produce an attribution result, a confidence score, and transparency label for the user. If the score strongly detects AI generation, the result is `likely_ai`. If the score strongly detects human writing,the result is `likely_human`. If the score is near the decision boundary, the result is `uncertain`.

After classification, an audit log entry of the decision is documented into the system. The log entry includes content ID, submitted text, both signal scores, combined score, attribution result, confidence score, transparency label, timestamp, and current status.

If a user disagrees with the classification decision, they can submit an appeal through `POST /appeal`. The appeal must include the content ID and the user’s reasoning as to why the decision was incorrect. The system finds the original decision in the audit log, updates the status to `under_review`, and returns a confirmation message.

## Detection Signals

### Signal 1: LLM Attribution Classifier

**Tool:** Groq using llama-3.3-70b-versatile

**What it measures:**

- Semantic coherence
- Whether the wording feels too generic or polished
- How natural the text is
- How specific the details are
- Whether the text sounds scripted or more like something a person would actually write

**Why is this signal useful?:**
The LLM can be affected by what the text is about and how it’s written. A clean, polished human essay might seem AI-written, while messy AI writing might seem human.

**Blind spot:**
The LLM can be influenced by the topic, genre, or style of the text. A polished human-written essay may look AI-like, while messy AI-generated text may look human.

### Signal 2: Stylometric Heuristics

**Tool:** Python stylometric analysis

**What it measures:**

- How often certain words repeat
- How much punctuation is used
- Average sentence length

**Why is this signal useful?:**
Stylometric features measure the structure of writing, not its meaning. That gives the system a separate signal that uses different reasoning than the LLM classifier.

**Blind spot:**
Stylometric heuristics cannot tell you the meaning of a text. Some people naturally write in a consistent, polished, or repetitive way. And AI generated text can always be edited to sound more human.

## API Surface

**`POST /submit`**
Receives a text submission and returns an attribution decision

**Request Data:**
{
  "user_id": "user_123",

  "content": "Submitted text goes here."
}

**Response Data:**
{
  "content_id": "content_123",

  "attribution": "likely_ai | likely_human | uncertain",

  "confidence": 0.82,

  "transparency_label": "This content shows strong signals of AI generation. It may have been created or heavily assisted by AI. Creators may appeal this label if they believe it is incorrect.",

  "signals": {
    "llm_score": 0.87,

    "stylometric_score": 0.72
  },

  "status": "classified"
}

**`POST /appeal`**
Accepts an appeal for a previous classification.

**Request Data:**
{
  "content_id": "generated-content-id",
  "creator_reason": "I wrote this myself and can provide drafts or notes."
}

**Response Data:**
{
  "content_id": "generated-content-id",
  "status": "under_review",
  "message": "Appeal received and logged for review."
}


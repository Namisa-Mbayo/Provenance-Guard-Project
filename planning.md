# Provenance Guard Planning

## Architecture Narrative

When a user submits a piece of text, it goes through the `POST /submit` endpoint as a request. First, the submission is checked by the rate limiter to prevent spam. If the user is sending too many requests, the API returns a message about the rate limit.

If the request passes rate limiting, the API validates the input by checking if the text is long enough to analyze. If the text is too short or missing, the API returns an error message. If the input is valid, the text goes through two detection signals. The first signal is an LLM based attribution classifier using Groq. This signal evaluates the semantics of the writing, such as tone, coherence, and whether the text resembles AI generated writing. The second signal is a stylometric heuristic analyzer. This signal measures structural properties of the text, such as sentence length variance, type token ratio, and punctuation.

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
The LLM can be influenced by the topic, genre, or style of the text. A polished human-written essay could look like AI, while messy AI-generated text could look more human.

### Signal 2: Stylometric Heuristics

**Tool:** Python stylometric analysis

**What it measures:**

- How often certain words repeat
- How much punctuation is used
- Average sentence length

**Why is this signal useful?:**
Stylometric features measure the structure of writing, not its meaning. That gives the system a separate signal that uses different reasoning than the LLM classifier.

**Blind spot:**
Stylometric heuristics cannot tell you the meaning or intent of a text. Some people naturally write in a consistent, polished, or repetitive way. And AI generated text can always be edited to sound more human.

### Signal Combination

The system combines both signals into an one score.

`final_score = (0.65 * llm_score) + (0.35 * stylometric_score)`

I chose to give the LLM signal more weight because it can evaluate meaning, context, and tone. The stylometric signal is still useful because it catches writing patterns that the LLM might miss.

## Uncertainty representation

A score closer to 1 means the text looks more like AI writing. A score closer to 0 means it looks more like human writing. A score near 0.5 means the system is unsure. For example, a score of 0.51 should not be labeled `likely_ai`. It should be labeled `uncertain` because the score is too close to the middle. The uncertainty threshold will be between 0.25 to 0.74, because AI attribution can be unreliable and false positives can harm human users. The system needs a high enough score to be confident in its decison.

**Example:**

| Score | Result | Explanation |
| - | - | - |
| 0.12 | `likely_human` | Strong human-writing signals |
| 0.51 | `uncertain` | Mixed signals, near the middle |
| 0.62 | `uncertain` | Some AI-generation signals, but not enough for a confident label |
| 0.91 | `likely_ai` | Strong AI-generation signals |

## Transparency Label Design

The transparency label is the plain-language text shown to users on the platform.

| Label | Message |
| - | - |
| High Confidence AI Label | "This content shows strong signals of AI generation. It may have been created or heavily assisted by AI. Creators may appeal this label if they believe it is incorrect." |
| Uncertain Label | "This content has mixed signals. We cannot confidently determine whether it was human-written or AI-generated." |
| High Confidence Human Label | "This content shows strong signs of human writing. No major AI generation signals were detected." |

## Anticipated Edge Cases

### Edge Case 1: Polished human writing

Human-written text can be polished, structured, and grammatically correct. The LLM signal and stylometric signal might score it as `likely_ai` if it has little to no messiness or variation.

### Edge Case 3: AI text prompted to sound casual

AI-generated text can be prompted to include slang or informal phrasing. This could fool the stylometric signal into scoring it as `likely_human`.

## API Surface

**`POST /submit`**
Receives a text submission and returns an attribution decision

**Request Data:**
{
  "user_id": "user_123",

  "content": "User submitted text."
}

**Response Data:**
{
  "content_id": "content_123",

  "content": "User submitted text."

  "attribution": "likely_ai",

  "confidence": 0.82,

  "transparency_label": "This content shows strong signals of AI generation. It may have been created or heavily assisted by AI. Creators may appeal this decision if they believe it is incorrect.",

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
  "user_reason": "I wrote this myself and can provide drafts or notes."
}

**Response Data:**
{
  "content_id": "generated-content-id",
  "status": "under_review",
  "message": "Appeal received and logged for review."
}

**`GET /log`**
Returns recent audit log entries.

**Response Data:**
{
  "entries": [
    {
      "content_id": "content_abc123",
      "timestamp": "2026-07-02T12:00:00Z",
      "creator_id": "creator_123",
      "attribution": "likely_ai",
      "confidence": 0.82,
      "llm_score": 0.87,
      "stylometric_score": 0.72,
      "status": "under_review",
      "appeal": {
        "reason": "I wrote this myself and can provide drafts showing my process.",
        "timestamp": "2026-07-02T12:05:00Z"
      }
    }
  ]
}

**`GET /health`**
Returns a simple health check confirming that the API is running.

**Response Data:**
{
  "status": "ok"
}

## AI Tool

**Milestone 3: Submission endpoint and first signal**
I will give Claude my Architecture, API Surface, and Detection Signal sections. I will ask it to generate a Flask app skeleton with `POST /submit`, `GET /health`, input validation, and the first signal function using Groq.

I will verify the output by testing `POST /submit` directly before adding the second signal. I will check that the endpoint accepts text, rejects missing or too-short input, calls the LLM signal, and returns structured JSON.

**Milestone 4: Second signal and confidence scoring**
I will give the AI tool my Detection Signals, Signal Combination, Uncertainty Representation, and Architecture sections. I will ask it to implement the stylometric heuristic signal and the score  formula.

I will verify the output by testing `likely_human`, `likely_ai`, and `uncertain` text samples. I will check that a score near 0.5 produces the `uncertain` label instead of `likely_ai` or `likely_human`.

**Milestone 5: Production layer**
I will give the AI tool my Transparency Label Design, Appeals Workflow, API Surface, and Architecture sections. I will ask it to implement the transparency label function, audit logging, `POST /appeal`, `GET /log`, and rate limiting.

I will verify the output by testing all three transparency label variants, submitting an appeal, checking that the content status updates to `under_review`. Then, I will then confirm that the audit log contains three entries along with their signal scores, confidence, labels, and appeal information.

import os, re, json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__)

# Nebius Token Factory is OpenAI-compatible. Check the base URL and the exact
# NVIDIA Nemotron model ID in your Token Factory console and set them below.
client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
    api_key=os.getenv("NEBIUS_API_KEY"),
)
MODEL = os.getenv("NEBIUS_MODEL", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1")
MAX_CHARS = 12000


def ask(system, user, max_tokens=1500):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "detailed thinking off\n" + system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content.strip()


def get_notes():
    text = (request.get_json(silent=True) or {}).get("notes", "").strip()
    if len(text) < 30:
        return None, ("Please paste at least a few lines of notes.", 400)
    return text[:MAX_CHARS], None


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.post("/api/summarise")
def summarise():
    notes, err = get_notes()
    if err:
        return jsonify(error=err[0]), err[1]
    try:
        out = ask(
            "You help students revise at the last minute. Summarise the notes as "
            "short bullet points grouped by topic, then add a 'Key terms' list "
            "and a 'Remember this' line. Use only the given notes.",
            notes,
        )
        return jsonify(summary=out)
    except Exception as e:
        return jsonify(error=f"AI request failed: {e}"), 502


@app.post("/api/quiz")
def quiz():
    notes, err = get_notes()
    if err:
        return jsonify(error=err[0]), err[1]
    n = min(max(int((request.get_json() or {}).get("count", 5)), 1), 10)
    try:
        out = ask(
            f"Create {n} multiple-choice questions from the notes. Reply with ONLY "
            'a JSON array like [{"q":"...","options":["A","B","C","D"],'
            '"answer":0,"why":"..."}]. "answer" is the index (0-3) of the correct '
            "option. No text outside the JSON.",
            notes,
            max_tokens=2500,
        )
        m = re.search(r"\[.*\]", out, re.S)
        questions = json.loads(m.group(0))
        return jsonify(questions=questions)
    except Exception as e:
        return jsonify(error=f"Could not build the quiz, try again: {e}"), 502


if __name__ == "__main__":
    app.run(debug=True)

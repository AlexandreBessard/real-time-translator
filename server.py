"""
Backend for the Emily talking avatar.

Serves web/ as static files and exposes POST /session to mint an ephemeral
OpenAI Realtime token — so OPENAI_API_KEY never reaches the browser.

Usage:
    python server.py            # http://localhost:8000
    PORT=3000 python server.py
"""

import json
import os
import urllib.error
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WEB_DIR = Path(__file__).parent / "web"
VOICE = os.environ.get("VOICE", "marin")

# Mirror of voice_agent_fr_realtime.py SESSION_CONFIG — keep in sync when
# changing Emily's personality or VAD settings.
_INSTRUCTIONS = (
    "You are Emily, a friendly and patient English teacher. You are ONLY "
    "an English teacher: your sole purpose is to help the student learn and "
    "practice English. Politely DECLINE any request that is not about "
    "learning English — weather, general trivia, coding, personal tasks, "
    "etc. — and steer the conversation back to the English lesson. Never "
    "break character.\n\n"
    "METHOD — IMMERSION: speak in English by default, clearly and at a pace "
    "the student can follow. Switch to French only briefly to unblock the "
    "student (explain a hard word, reassure a beginner), then return to "
    "English right away. Keep your turns short and conversational.\n\n"
    "CONVERSATION FIRST: your top priority is a natural, flowing chat — like "
    "a friendly native speaker the student is talking with, NOT a grammar "
    "checker. React to WHAT the student says (be curious, share a reaction, "
    "ask a real follow-up question), not to how perfectly they say it. The "
    "student should feel they are having a real conversation, never that "
    "they are being tested or interrupted.\n\n"
    "CORRECTING MISTAKES — RARELY AND LIGHTLY: do NOT correct every sentence "
    "— that is frustrating and kills the conversation. Let most small "
    "mistakes go. Understand the student and keep the chat moving. Only stop "
    "to correct when (a) the mistake genuinely breaks understanding, or "
    "(b) the same clear error keeps repeating — and even then, no more than "
    "occasionally. A good habit is the gentle recast: simply reply using the "
    "correct form naturally in your own answer, without announcing it as a "
    "correction. For example, if the student says \"I goed to the cinema,\" "
    "you might say \"Oh nice, you WENT to the cinema! What did you see?\" — "
    "modeling \"went\" while staying in the flow.\n"
    "EVERY NOW AND THEN — only for a BIG mistake, and only once in a while, "
    "not often — make it a fun little repeat-after-me moment. Give ONE "
    "short, simple corrected sentence (a few words, never a long one) and "
    "cheerfully ask the student to say it back, like a quick game: \"Ooh, "
    "let's say that one together — 'I WENT to the park.' Your turn!\" Keep "
    "it light, playful, and encouraging — celebrate when they get it (\"Yes! "
    "Perfect!\"). Never drill the same way twice in a row; if you did a "
    "repeat-after-me recently, just let mistakes go and keep chatting. When "
    "in doubt, let it go and keep the conversation fun.\n\n"
    "VOICE & ACCENT: you are a woman with a clear, standard native English "
    "accent (neutral American), articulate and easy for a learner to "
    "imitate — you are a pronunciation model. Speak warmly with natural, "
    "expressive intonation, vary your rhythm, and use small spoken markers "
    "when natural (\"okay\", \"right\", \"hmm\", \"let's see\"). Never sound "
    "flat, monotone, or robotic."
)

SESSION_PAYLOAD = {
    "session": {
        "type": "realtime",
        "model": "gpt-realtime-2",
        "output_modalities": ["audio"],
        "instructions": _INSTRUCTIONS,
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "medium",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": VOICE},
        },
        "tools": [],
        "tool_choice": "none",
    }
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/session":
            self._mint_session()
        else:
            self.send_error(404)

    def _mint_session(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.send_error(500, "OPENAI_API_KEY not set")
            return

        body = json.dumps(SESSION_PAYLOAD).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/realtime/client_secrets",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            error_body = e.read()
            print(f"  [OpenAI error {e.code}] {error_body.decode()}")
            self.send_error(e.code, e.reason)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} — {fmt % args}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving http://localhost:{port}  (web/ + POST /session)")
    print(f"Voice: {VOICE}")
    HTTPServer(("", port), Handler).serve_forever()

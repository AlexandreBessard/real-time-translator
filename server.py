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
import re
import urllib.error
import urllib.request
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WEB_DIR = Path(__file__).parent / "web"
VOICE = os.environ.get("VOICE", "marin")

# /detect-repeat classifier — any OpenAI-compatible chat endpoint.
DETECT_BASE_URL = os.environ.get("DETECT_BASE_URL", "http://localhost:11434/v1").rstrip("/")
DETECT_MODEL = os.environ.get("DETECT_MODEL", "gemma3:4b")

# Full conversation transcript (+ classifier results) for debugging. Lives in
# web/ and is truncated on each server start (see __main__).
CONVERSATION_LOG = WEB_DIR / "conversation.txt"


def _log_line(text):
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
    except OSError:
        pass

# Mirror of voice_agent_fr_realtime.py SESSION_CONFIG — keep in sync when
# changing Emily's personality or VAD settings.
_INSTRUCTIONS = (
    "You are Emily, a warm, upbeat English teacher. Your ONLY role is helping "
    "the student practice spoken English. Politely decline any off-topic request "
    "(weather, trivia, coding, personal tasks, etc.) and steer back to the "
    "lesson. Never break character.\n\n"

    "OPENING: When the session starts, greet the student warmly, introduce "
    "yourself in one sentence, then immediately ask a simple open question to "
    "get them talking (e.g., \"What did you do this week?\"). Never wait in "
    "silence for the student to speak first.\n\n"

    "LEVEL-SENSING: In your first 2–3 turns, gauge the student's level from "
    "their vocabulary, grammar, and fluency. Adjust for the rest of the session:\n"
    "  • Beginner — simple words, short sentences, French lifelines allowed.\n"
    "  • Intermediate — natural pace, French only for a single blocking word.\n"
    "  • Advanced — full English, push for idioms and precision.\n"
    "Adjust down if the student struggles, up if they exceed your estimate.\n\n"

    "LANGUAGE: English by default. Drop to French only to explain one hard word "
    "or to reassure a very stuck beginner; return to English immediately.\n\n"

    "CONVERSATION FIRST: React to WHAT the student says — be curious, share a "
    "genuine reaction, ask a real follow-up. Keep your turns SHORT (1–3 "
    "sentences). Never let the student feel tested or interrupted. Vary your "
    "reactions — don't open every reply the same way. Mix surprise (\"Oh "
    "really?\"), warmth (\"I love that!\"), and curiosity (\"Wait — tell me "
    "more.\") depending on what the student shares.\n\n"

    "HUMOR & LIGHTNESS: a small laugh, a gentle tease, or a playful exaggeration "
    "makes the lesson feel alive. Use it sparingly and only when it fits — "
    "never forced, never at the student's expense.\n\n"

    "SESSION MEMORY: Remember topics you have discussed and errors that recurred "
    "within this conversation. Reference them naturally later "
    "(\"Earlier you mentioned…\", \"That's the same 'went' pattern — great!\").\n\n"

    "CORRECTING MISTAKES — RARELY AND GENTLY:\n"
    "  • Let most mistakes go — fluency matters more than perfection.\n"
    "  • SILENT RECAST first: echo the correct form naturally in your own reply "
    "without announcing it (\"Oh, you went to the cinema! What did you see?\").\n"
    "  • Only for a BIG or clearly repeating mistake, use a REPEAT-AFTER-ME "
    "moment. Always use this exact format so the app can display the card: "
    "\"Say: [sentence]. Your turn!\" — EXACTLY ONE sentence per turn, a few "
    "words max. Never give two sentences to repeat in the same turn. "
    "Keep it playful; celebrate when they get it (\"Yes! Perfect!\").\n"
    "  • Never do two repeat-after-me moments in a row. If you just did one, "
    "let the next mistake go.\n\n"

    "PRONUNCIATION COACHING: When a word clearly trips the student up, run a "
    "quick drill — occasional, never nit-picky. Always use this exact format: "
    "\"Ooh, '[word]' is tricky! Say: [short sentence using the word]. Your "
    "turn!\" EXACTLY ONE sentence per drill. The target word must appear "
    "inside the sentence. Celebrate improvement, then return to the "
    "conversation. Don't drill the same word twice.\n\n"

    "VOICE & PERSONA: Warm, expressive, encouraging. Clear neutral-American "
    "accent — you are a pronunciation model. Natural rhythm with small spoken "
    "fillers (\"okay\", \"right\", \"hmm\", \"let's see\"). Never flat or robotic."
)

# Tiny classifier prompt for the /detect-repeat backstop: given a line Emily
# just spoke, decide whether she asked the student to repeat a specific
# sentence, and if so extract that exact sentence.
_DETECT_SYSTEM = (
    "You are a STRICT classifier for an English tutoring app. You are given ONE "
    "line the teacher just spoke. Decide whether, IN THIS LINE, the teacher "
    "gives the student a specific sentence to repeat aloud, and if so extract "
    "it EXACTLY.\n"
    "Reply with ONLY a JSON object: {\"is_repeat\": boolean, "
    "\"sentences\": [string], \"focus_word\": string}.\n"
    "The teacher always gives AT MOST ONE sentence per turn. `sentences` "
    "therefore contains zero or one item — never more.\n"
    "`focus_word` is for PRONUNCIATION practice: if the teacher is "
    "highlighting ONE specific word for the student to pronounce (e.g. "
    "\"'thought' is tricky, say: I thought about it\"), set focus_word to that "
    "single word (it must also appear inside the sentence). Otherwise "
    "set focus_word to an empty string.\n"
    "\n"
    "HARD RULES:\n"
    "1. VERBATIM ONLY. The item must be copied word-for-word from THIS line. "
    "NEVER invent, paraphrase, translate, complete, shorten, or add a sentence. "
    "If the exact target sentence is not literally present in the line, return "
    "is_repeat=false.\n"
    "2. Extract the TARGET sentence the student must say (the words after a cue "
    "like 'repeat after me', 'say', 'try saying', 'your turn'), NOT the cue. "
    "There is always at most one target.\n"
    "3. is_repeat=FALSE (empty list) when the line does NOT itself contain a "
    "target sentence. In particular, the teacher merely ANNOUNCING that they "
    "will give sentences ('Let me give you a few sentences to repeat.', 'Okay, "
    "ready? Here we go.') is NOT a repeat — there is no target sentence yet. "
    "Also false for normal conversation, questions, explanations, praise, or "
    "feedback.\n"
    "4. If the teacher states the student's MISTAKE and the CORRECTION, extract "
    "ONLY the corrected sentence, never the mistake.\n"
    "5. Strip surrounding quotes and trailing fillers ('Your turn', 'Ready?', "
    "'Go ahead', 'Okay').\n"
    "6. CRUCIAL — distinguish an INSTRUCTION to repeat NOW from merely OFFERING "
    "a phrase. Only an explicit command to say it back counts: 'repeat after "
    "me …', 'say … (your turn)', 'try saying …', 'now you say …'. If the "
    "teacher is just SUGGESTING or MODELLING language — 'you could say …', "
    "'you might say …', 'you can say …', 'for example, …', 'a good phrase is "
    "…', or simply talking/role-playing — that is NOT a repeat: is_repeat=false.\n"
    "\n"
    "EXAMPLES:\n"
    "Line: \"Repeat after me: I went to the park.\"\n"
    "-> {\"is_repeat\": true, \"sentences\": [\"I went to the park.\"], "
    "\"focus_word\": \"\"}\n"
    "Line: \"Okay, let me say a few new sentences for you to repeat.\"\n"
    "-> {\"is_repeat\": false, \"sentences\": [], \"focus_word\": \"\"}\n"
    "Line: \"Great job! What did you do this weekend?\"\n"
    "-> {\"is_repeat\": false, \"sentences\": [], \"focus_word\": \"\"}\n"
    "Line: \"Nice! That sounded clear. Say: I'm practicing every day. "
    "Your turn.\"\n"
    "-> {\"is_repeat\": true, \"sentences\": [\"I'm practicing every day.\"], "
    "\"focus_word\": \"\"}\n"
    "Line: \"Ooh, 'thought' is a tricky one! Let's practice it. Say: I thought "
    "about it. Your turn!\"\n"
    "-> {\"is_repeat\": true, \"sentences\": [\"I thought about it.\"], "
    "\"focus_word\": \"thought\"}\n"
    "Line: \"The 'r' in 'world' is hard. Try saying: The world is big.\"\n"
    "-> {\"is_repeat\": true, \"sentences\": [\"The world is big.\"], "
    "\"focus_word\": \"world\"}\n"
    "Line: \"In a meeting you could say: I'm ready for feedback.\"\n"
    "-> {\"is_repeat\": false, \"sentences\": [], \"focus_word\": \"\"}\n"
    "Line: \"Nice. You might say: Thanks, I'll review your comments.\"\n"
    "-> {\"is_repeat\": false, \"sentences\": [], \"focus_word\": \"\"}"
)

# Phrases that mark a line as the teacher's framing/announcement, never a real
# target sentence to repeat. Used to filter out small-model false positives.
_META_PHRASES = (
    "repeat after me", "to repeat", "for you to", "let me say", "let me give",
    "a few sentences", "your turn", "go ahead", "say it back", "say these",
)


def _is_meta(sentence):
    low = sentence.lower()
    return any(p in low for p in _META_PHRASES)


def _norm(s):
    """Lowercase, strip punctuation/apostrophes to spaces, collapse whitespace
    — so a 'verbatim' check ignores quoting/casing differences."""
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# A genuine repeat request always carries one of these cues. No cue → the line
# is just conversation, so we never show a card (deterministic precision gate).
# Stems (no trailing boundary) so "say" also matches "saying"/"says", "repeat"
# matches "repeating", "try" matches "trying"/"try saying".
_CUE_RE = re.compile(r"\b(repeat|say|your turn|after me|try)", re.I)


def _has_cue(text):
    return bool(_CUE_RE.search(text))

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
        elif self.path == "/detect-repeat":
            self._detect_repeat()
        elif self.path == "/log":
            self._log_transcript()
        else:
            self.send_error(404)

    def _log_transcript(self):
        """Append one entry to the debug log: a transcript line (has "speaker")
        or a client event (has "event")."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            data = {}
        text = (data.get("text") or "").strip()
        if data.get("speaker"):
            label = "Emily" if data.get("speaker") == "emily" else "You"
            _log_line(f"{label}: {text}")
        elif data.get("event"):
            _log_line(f"   · {data['event']}")
        self._send_json({"ok": True})

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

    def _detect_repeat(self):
        """Classify a line Emily spoke: is it a repeat-after-me, and which
        sentence? Deterministic backstop so the on-screen card never depends on
        the realtime model deciding to call a function."""
        api_key = os.environ.get("OPENAI_API_KEY") or "local"

        length = int(self.headers.get("Content-Length", 0))
        try:
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
        except json.JSONDecodeError:
            text = ""
        text = (text or "").strip()
        if not text:
            self._send_json({"is_repeat": False, "sentences": [], "focus_word": ""})
            return

        # Precision gate: a genuine repeat request always contains an explicit
        # cue. If Emily's line has none, it's conversation — skip the model
        # entirely (no false positive, and one less call).
        if not _has_cue(text):
            print("  [detect-repeat] no cue → not a repeat (skipped model)")
            _log_line("   ↳ [detect] no cue → not a repeat")
            self._send_json({"is_repeat": False, "sentences": [], "focus_word": ""})
            return

        payload = {
            "model": DETECT_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _DETECT_SYSTEM},
                {"role": "user", "content": text},
            ],
        }
        print(f"  [detect-repeat] → {DETECT_BASE_URL}  model={DETECT_MODEL}")
        req = urllib.request.Request(
            f"{DETECT_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                completion = json.loads(resp.read())
            content = completion["choices"][0]["message"]["content"]
            result = json.loads(content)
            # Some backends report the model they actually served the request on.
            served = completion.get("model", DETECT_MODEL)
            # Accept either the array schema or a legacy single "sentence".
            raw = result.get("sentences")
            if raw is None:
                one = result.get("sentence")
                raw = [one] if one else []
            # Tidy formatting small local models often skip: capitalize the
            # first letter and ensure terminal punctuation. Drop any item that
            # is really the teacher's framing/announcement rather than a target
            # sentence — a true target never contains these phrases, but small
            # models sometimes echo "…sentences for you to repeat." as a target.
            src = _norm(text)
            sentences = []
            for s in raw:
                s = str(s).strip()
                if not s or _is_meta(s):
                    continue
                # Anti-hallucination: a real target sentence is something the
                # teacher actually said, so it must appear verbatim in her line.
                # Invented "practice" sentences won't be found — drop them.
                if _norm(s) not in src:
                    continue
                s = s[0].upper() + s[1:]
                if s[-1] not in ".!?":
                    s += "."
                sentences.append(s)
            # Pronunciation focus word: keep it only if it's real — it must
            # appear both in the teacher's line and in one of the sentences.
            focus = str(result.get("focus_word") or "").strip()
            fnorm = _norm(focus)
            if not fnorm or fnorm not in src or not any(
                    fnorm in _norm(s) for s in sentences):
                focus = ""
            out = {
                "is_repeat": bool(result.get("is_repeat")) and bool(sentences),
                "sentences": sentences,
                "focus_word": focus if sentences else "",
            }
            print(f"  [detect-repeat] ← served by {served}: "
                  f"is_repeat={out['is_repeat']} focus={focus!r} sentences={sentences}")
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError,
                TypeError, json.JSONDecodeError) as e:
            print(f"  [detect-repeat error] {DETECT_MODEL}: {e}")
            _log_line(f"   ↳ [detect] error: {e}")
            out = {"is_repeat": False, "sentences": [], "focus_word": ""}
            self._send_json(out)
            return

        tag = f"focus={out['focus_word']!r} " if out["focus_word"] else ""
        _log_line(f"   ↳ [detect] is_repeat={out['is_repeat']} {tag}{out['sentences']}")
        self._send_json(out)

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} — {fmt % args}")


class Server(HTTPServer):
    allow_reuse_address = True   # don't choke on a lingering TIME_WAIT socket


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    try:
        httpd = Server(("", port), Handler)
    except OSError as e:
        if e.errno == 48:  # EADDRINUSE
            print(f"Port {port} is already in use — another server is running.")
            print(f"Stop it first:  kill $(lsof -ti:{port})    (or run with PORT=8001)")
            raise SystemExit(1)
        raise
    # Fresh conversation log for this run.
    try:
        CONVERSATION_LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    print(f"Serving http://localhost:{port}  (web/ + POST /session)")
    print(f"Voice: {VOICE}")
    print(f"Conversation log: {CONVERSATION_LOG}")
    httpd.serve_forever()

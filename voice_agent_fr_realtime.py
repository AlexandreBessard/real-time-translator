"""
English-teacher voice agent — live conversational mode via OpenAI Realtime API.

A patient, immersion-style English teacher (female, native English voice).
Speaks English by default, drops to French only to unblock the student, and
stays strictly on-topic: it declines anything that isn't an English lesson.

Architecture: speech-to-speech (gpt-realtime-2)
  - Mic audio streams continuously; server VAD handles turn detection
  - Model manages barge-in and interruptions natively
  - No tools — the agent is only an English teacher

Usage:
    python voice_agent_fr_realtime.py

Requirements:
    pip install websocket-client pyaudio numpy python-dotenv
"""

import base64
import json
import os
import queue
import threading
import time

import numpy as np
import pyaudio
import websocket
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24_000   # rate the Realtime API expects (in and out)
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4_800  # 200 ms at 24 kHz
PLAYBACK_TAIL_S = 0.4  # keep mic muted briefly after audio drains (device buffer)

# Echo handling.
#   Headphones (default): the mic never hears the agent, so we run FULL-DUPLEX —
#     the mic stays open during playback and the server's VAD + interrupt_response
#     handle barge-in natively. Just talk over the agent to stop it.
#   Open speakers: the mic WOULD hear the agent and interrupt itself, so set this
#     to True to mute the mic while the agent speaks (no voice barge-in then —
#     that needs acoustic echo cancellation).
ECHO_FROM_SPEAKERS = os.environ.get("ECHO_FROM_SPEAKERS", "") not in ("", "0", "false", "False")

# Input mic selection: a name substring (e.g. "snowball") or a numeric device index.
# Override from the shell, e.g.  INPUT_DEVICE="Blue Snowball" python voice_agent_fr_realtime.py
INPUT_DEVICE = os.environ.get("INPUT_DEVICE", "snowball")

# Output voice. We want a clearly FEMALE voice: "marin" is the newest, most
# natural GA voice and reads female. Other female-leaning options if you want
# to compare: "coral", "sage", "shimmer". Override from the shell, e.g.
#   VOICE=coral python voice_agent_fr_realtime.py
VOICE = os.environ.get("VOICE", "marin")

WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"

# ── Audio device helpers ──────────────────────────────────────────────────────

def find_input_device(pa: pyaudio.PyAudio, hint: str) -> int | None:
    """Resolve an input device index from a numeric index or a name substring.

    Returns None to fall back to the system default input device.
    """
    if not hint:
        return None
    if hint.strip().isdigit():
        return int(hint.strip())

    hint_low = hint.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0 and hint_low in info["name"].lower():
            return i

    print(f"[AVERTISSEMENT] Aucun micro nommé « {hint} » trouvé. Périphériques d'entrée :")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"   [{i}] {info['name']}")
    print("   → utilisation du micro par défaut.")
    return None


def resample_to_24k(pcm: bytes, src_rate: int, src_channels: int) -> bytes:
    """Downmix to mono and resample int16 PCM from src_rate to SAMPLE_RATE."""
    if not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if src_channels > 1:
        samples = samples.reshape(-1, src_channels).mean(axis=1)

    if src_rate != SAMPLE_RATE:
        dst_n = int(round(samples.size * SAMPLE_RATE / src_rate))
        if dst_n <= 0:
            return b""
        # Linear interpolation — fine for speech; handles non-integer ratios
        # like the Snowball's 44.1 kHz → 24 kHz.
        x_old = np.arange(samples.size, dtype=np.float32)
        x_new = np.linspace(0, samples.size - 1, dst_n, dtype=np.float32)
        samples = np.interp(x_new, x_old, samples)

    return samples.astype(np.int16).tobytes()


# ── Session config sent after session.created ─────────────────────────────────

SESSION_CONFIG = {
    "type": "realtime",
    "model": "gpt-realtime-2",
    "output_modalities": ["audio"],
    "instructions": (
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
    ),
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            # semantic_vad decides end-of-turn from MEANING, not a silence timer:
            # it waits through mid-thought pauses ("euh… je voudrais…") yet
            # replies promptly once you've clearly finished. eagerness tunes the
            # speed/patience balance — "high" = snappier, "low" = more patient,
            # "medium" (auto) = balanced.
            "turn_detection": {
                "type": "semantic_vad",
                "eagerness": "medium",
                "create_response": True,
                "interrupt_response": True,
            },
            # far_field suits a laptop/built-in mic; use "near_field" for a headset.
            "noise_reduction": {"type": "far_field"},
            "transcription": {"model": "whisper-1"},
        },
        "output": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "voice": VOICE,
        },
    },
    # No tools — this agent is ONLY an English teacher and never calls out.
    "tools": [],
    "tool_choice": "none",
}


# ── Agent ─────────────────────────────────────────────────────────────────────

class FrenchVoiceAgent:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.playback_queue: queue.Queue[bytes | None] = queue.Queue()
        self.ws: websocket.WebSocket | None = None
        self._stop = threading.Event()
        # Accumulate streaming tool-call arguments keyed by call_id
        self._pending_calls: dict[str, dict] = {}
        # Half-duplex echo suppression state
        self._speaking = threading.Event()   # set while a response is generating
        self._play_until = 0.0               # monotonic time the speaker goes silent
        # True once we've answered tool call(s) and owe the model a follow-up turn
        self._needs_response = False

    # ── WebSocket ──────────────────────────────────────────────────────────────

    def _connect(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — check your .env file.")
        self.ws = websocket.WebSocket()
        self.ws.connect(
            WS_URL,
            header=[
                f"Authorization: Bearer {api_key}",
                "OpenAI-Safety-Identifier: voice-agent-fr",
            ],
        )

    def _send(self, event: dict):
        self.ws.send(json.dumps(event))

    # ── Tool dispatch ──────────────────────────────────────────────────────────

    def _dispatch_tool(self, call_id: str, name: str, args_json: str):
        # This agent exposes no tools, so this path should never be reached.
        # Kept as a safe fallback in case the model ever invents a tool call.
        result = f"Unknown tool: {name}"
        print(f"  → {result}")

        # Return the result to the model. Don't request a new response here —
        # a single turn may contain several parallel tool calls, so we trigger
        # exactly one response.create after response.done (see _receive_events),
        # otherwise we'd hit "conversation_already_has_active_response".
        self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        })
        self._needs_response = True

    # ── Receive thread ─────────────────────────────────────────────────────────

    def _receive_events(self):
        agent_buf = ""

        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
            except Exception:
                break

            if not raw:
                break

            event = json.loads(raw)
            etype = event.get("type", "")

            if etype == "session.created":
                self._send({"type": "session.update", "session": SESSION_CONFIG})
                print("Lesson ready — say hello in English!\n")

            elif etype == "session.updated":
                print(f"[DEBUG session] {json.dumps(event.get('session', {}), indent=2)}")

            elif etype == "response.created":
                self._speaking.set()

            elif etype == "response.done":
                self._speaking.clear()
                # All tool outputs for this turn are now submitted — ask the
                # model for its spoken follow-up exactly once.
                if self._needs_response:
                    self._needs_response = False
                    self._send({"type": "response.create"})

            elif etype == "input_audio_buffer.speech_started":
                # Barge-in: stop the agent's audio at once and open the mic fully
                # so the rest of the user's interruption flows through.
                self._flush_playback()
                self._speaking.clear()

            elif etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"\n[Vous]  {transcript}")

            elif etype == "response.output_audio.delta":
                pcm = base64.b64decode(event["delta"])
                # Advance the playback clock so we know when the speaker will
                # actually fall silent. PortAudio buffers faster than realtime
                # and response.done fires when generation ends (not playback),
                # so an empty queue does NOT mean the audio finished playing.
                secs = len(pcm) / 2 / SAMPLE_RATE  # 2 bytes per int16 mono sample
                self._play_until = max(self._play_until, time.monotonic()) + secs
                self.playback_queue.put(pcm)

            elif etype == "response.output_audio_transcript.delta":
                agent_buf += event.get("delta", "")
                print(f"\r[Agent] {agent_buf}", end="", flush=True)

            elif etype == "response.output_audio_transcript.done":
                print()
                agent_buf = ""

            # Tool call: model declares which function it wants to call
            elif etype == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    cid = item["call_id"]
                    self._pending_calls[cid] = {"name": item["name"], "args": ""}

            # Tool call: stream the JSON arguments
            elif etype == "response.function_call_arguments.delta":
                cid = event.get("call_id", "")
                if cid in self._pending_calls:
                    self._pending_calls[cid]["args"] += event.get("delta", "")

            # Tool call: arguments complete → execute locally
            elif etype == "response.function_call_arguments.done":
                cid = event.get("call_id", "")
                if cid in self._pending_calls:
                    tc = self._pending_calls.pop(cid)
                    args_str = event.get("arguments", "{}")
                    print(f"\n[Outil] {tc['name']}({args_str})")
                    self._dispatch_tool(cid, tc["name"], args_str)

            elif etype == "error":
                print(f"\n[ERREUR] {event.get('error', event)}")

        self._stop.set()

    # ── Capture thread ─────────────────────────────────────────────────────────

    def _open_input_stream(self):
        """Open the chosen input device, returning (stream, rate, channels, name)."""
        dev_index = find_input_device(self.pa, INPUT_DEVICE)
        if dev_index is not None:
            info = self.pa.get_device_info_by_index(dev_index)
        else:
            info = self.pa.get_default_input_device_info()
        name = info["name"]
        native_rate = int(info.get("defaultSampleRate", SAMPLE_RATE)) or SAMPLE_RATE
        max_ch = max(1, int(info.get("maxInputChannels", 1)))

        # Try 24 kHz mono first (no resampling). Fall back to the device's native
        # rate/channels — most reliable on macOS — and we resample ourselves.
        for rate, ch in [(SAMPLE_RATE, 1), (native_rate, 1), (native_rate, min(max_ch, 2))]:
            try:
                stream = self.pa.open(
                    format=FORMAT,
                    channels=ch,
                    rate=rate,
                    input=True,
                    input_device_index=dev_index,
                    frames_per_buffer=int(rate * 0.2),  # ~200 ms
                )
                return stream, rate, ch, name
            except Exception:
                continue
        raise RuntimeError(f"Impossible d'ouvrir le micro « {name} ».")

    def _capture_audio(self):
        stream, rate, channels, name = self._open_input_stream()
        read_frames = int(rate * 0.2)
        resampling = rate != SAMPLE_RATE or channels != 1
        suffix = f" ({rate} Hz → {SAMPLE_RATE} Hz)" if resampling else ""
        print(f"Micro : {name}{suffix}")
        print("Parlez à tout moment. Ctrl+C pour quitter.\n")
        try:
            while not self._stop.is_set():
                # Always drain the device buffer.
                data = stream.read(read_frames, exception_on_overflow=False)
                if resampling:
                    data = resample_to_24k(data, rate, channels)

                # FULL-DUPLEX (headphones): keep the mic open during playback so
                # the server's VAD hears you talk over the agent and cancels its
                # turn (interrupt_response). On open SPEAKERS that would let the
                # agent hear itself, so there we mute the mic while it speaks.
                if ECHO_FROM_SPEAKERS and (
                    self._speaking.is_set()
                    or time.monotonic() < self._play_until + PLAYBACK_TAIL_S
                ):
                    continue

                self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode(),
                })
        except Exception:
            pass
        finally:
            stream.stop_stream()
            stream.close()

    # ── Playback thread ────────────────────────────────────────────────────────

    def _play_audio(self):
        stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )
        try:
            while True:
                chunk = self.playback_queue.get()
                if chunk is None:  # shutdown sentinel
                    break
                stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def _flush_playback(self):
        """Discard all queued audio (called on barge-in)."""
        self._play_until = 0.0  # speaker about to go silent → let the mic reopen
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        print("=== English Teacher — Live Conversation ===")
        print("Connexion à OpenAI Realtime…")
        self._connect()

        receive_thread = threading.Thread(target=self._receive_events, daemon=True)
        capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        playback_thread = threading.Thread(target=self._play_audio, daemon=True)

        receive_thread.start()
        playback_thread.start()
        capture_thread.start()

        try:
            capture_thread.join()
        except KeyboardInterrupt:
            print("\nFermeture…")
        finally:
            self._stop.set()
            self.playback_queue.put(None)  # unblock playback thread
            receive_thread.join(timeout=5)
            playback_thread.join(timeout=5)
            try:
                self.ws.close()
            except Exception:
                pass
            try:
                self.pa.terminate()
            except Exception:
                pass
            print("Au revoir !")


if __name__ == "__main__":
    FrenchVoiceAgent().run()

"""
Real-time translation using OpenAI's gpt-realtime-translate model.

Captures microphone audio, streams it to the translation API, and plays
back translated audio while printing both source and target transcripts.

Usage:
    python translator.py                          # auto-detect source, translate to Spanish
    python translator.py --source fr --target en  # French → English
    python translator.py --source en --target ja  # English → Japanese
    python translator.py --list-devices           # list audio devices

Requirements:
    pip install websocket-client pyaudio python-dotenv
"""

import argparse
import base64
import json
import os
import queue
import struct
import threading

import pyaudio
import websocket
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4800  # 200ms of audio at 24kHz

WS_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


def list_audio_devices():
    pa = pyaudio.PyAudio()
    print(f"\n{'ID':<4} {'Name':<45} {'In':<5} {'Out'}")
    print("-" * 65)
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        print(f"{i:<4} {d['name'][:44]:<45} {int(d['maxInputChannels']):<5} {int(d['maxOutputChannels'])}")
    pa.terminate()


class RealtimeTranslator:
    def __init__(self, target_language: str, source_language: str | None = None, input_device: int | None = None, output_device: int | None = None):
        self.target_language = target_language
        self.source_language = source_language
        self.input_device = input_device
        self.output_device = output_device
        self.pa = pyaudio.PyAudio()
        self.playback_queue: queue.Queue[bytes] = queue.Queue()
        self.ws: websocket.WebSocket | None = None
        self.closing = False
        self._stop = threading.Event()

    # ── WebSocket ──────────────────────────────────────────────────────────

    def connect(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self.ws = websocket.WebSocket()
        self.ws.connect(
            WS_URL,
            header=[
                f"Authorization: Bearer {api_key}",
                "OpenAI-Safety-Identifier: test-client",
            ],
        )

    def configure_session(self):
        self.ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"output": {"language": self.target_language}}},
        }))

    def send_audio(self, pcm_bytes: bytes):
        payload = base64.b64encode(pcm_bytes).decode()
        self.ws.send(json.dumps({
            "type": "session.input_audio_buffer.append",
            "audio": payload,
        }))

    def close_session(self):
        if self.closing:
            return
        self.closing = True
        try:
            self.ws.send(json.dumps({"type": "session.close"}))
        except Exception:
            pass

    # ── Event loop (runs in its own thread) ───────────────────────────────

    def _receive_events(self):
        source_buf = ""
        target_buf = ""

        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
            except Exception:
                break

            event = json.loads(raw)
            etype = event.get("type")

            if etype == "session.output_audio.delta":
                audio_bytes = base64.b64decode(event["delta"])
                self.playback_queue.put(audio_bytes)

            elif etype == "session.output_transcript.delta":
                target_buf += event["delta"]
                print(f"\r[TARGET] {target_buf}", end="", flush=True)

            elif etype == "session.input_transcript.delta":
                source_buf += event["delta"]
                print(f"\n[SOURCE] {source_buf}", end="", flush=True)

            elif etype == "session.closed":
                print("\n[session closed]")
                self.playback_queue.put(None)  # sentinel to stop playback
                break

            elif etype == "error":
                print(f"\n[ERROR] {event}")
                self.playback_queue.put(None)
                break

        self._stop.set()

    # ── Audio capture (runs in its own thread) ────────────────────────────

    def _capture_audio(self):
        stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=CHUNK,
        )
        print(f"Capturing audio... speak now. Press Ctrl+C to stop.\n")
        try:
            while not self._stop.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.send_audio(data)
        except Exception:
            pass
        finally:
            stream.stop_stream()
            stream.close()

    # ── Audio playback (runs in its own thread) ───────────────────────────

    def _play_audio(self):
        stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=CHUNK,
        )
        try:
            while True:
                chunk = self.playback_queue.get()
                if chunk is None:  # sentinel
                    break
                stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    # ── Public API ────────────────────────────────────────────────────────

    def run(self):
        src = self.source_language or "auto"
        print(f"Connecting to OpenAI realtime translation ({src} → {self.target_language})...")
        self.connect()
        self.configure_session()
        print("Connected.\n")

        receive_thread = threading.Thread(target=self._receive_events, daemon=True)
        capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        playback_thread = threading.Thread(target=self._play_audio, daemon=True)

        receive_thread.start()
        playback_thread.start()
        capture_thread.start()

        try:
            capture_thread.join()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._stop.set()
            self.close_session()
            receive_thread.join(timeout=10)
            playback_thread.join(timeout=10)
            self.ws.close()
            self.pa.terminate()
            print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Real-time audio translation via OpenAI")
    parser.add_argument("--source", "-s", default=None, help="Source language code, e.g. fr (auto-detect if omitted)")
    parser.add_argument("--target", "-t", default="es", help="Target language code (default: es)")
    parser.add_argument("--input-device", "-i", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", "-o", type=int, default=None, help="Output device index")
    parser.add_argument("--list-devices", "-l", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    translator = RealtimeTranslator(
        target_language=args.target,
        source_language=args.source,
        input_device=args.input_device,
        output_device=args.output_device,
    )
    translator.run()


if __name__ == "__main__":
    main()
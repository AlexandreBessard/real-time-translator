"""
French voice agent with a mock weather tool, built with the OpenAI Agents SDK.

Architecture: chained VoicePipeline (STT → text agent → TTS)
  1. Microphone captures a spoken question in French
  2. Whisper transcribes it to text
  3. GPT-4o-mini reasons and calls get_weather if needed
  4. TTS converts the French answer back to speech

Usage:
    python voice_agent_fr.py

Requirements:
    pip install openai-agents pyaudio numpy python-dotenv
"""

import asyncio
import os

import numpy as np
import pyaudio
from dotenv import load_dotenv

from agents import Agent, function_tool, set_tracing_disabled
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.events import VoiceStreamEventAudio

load_dotenv()
set_tracing_disabled(True)

SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK = 4_800          # 200 ms at 24 kHz
RECORD_SECONDS = 5

# ── Mock weather data ─────────────────────────────────────────────────────────

_WEATHER_DB: dict[str, dict] = {
    "paris":     {"condition": "ensoleillé",             "temp": 22},
    "lyon":      {"condition": "nuageux",                "temp": 18},
    "marseille": {"condition": "partiellement nuageux",  "temp": 26},
    "bordeaux":  {"condition": "pluvieux",               "temp": 15},
    "nice":      {"condition": "ensoleillé",             "temp": 28},
    "toulouse":  {"condition": "venteux",                "temp": 20},
}


# ── Tool ─────────────────────────────────────────────────────────────────────

@function_tool
def get_weather(city: str) -> str:
    """Return the current weather for a French city."""
    data = _WEATHER_DB.get(city.lower().strip())
    if data:
        return (
            f"À {city.capitalize()} il fait {data['condition']} "
            f"avec {data['temp']} degrés Celsius."
        )
    return f"Je n'ai pas de données météo pour {city}."


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = Agent(
    name="Assistant Météo",
    instructions=(
        "Tu es un assistant vocal francophone. "
        "Réponds TOUJOURS en français, de façon concise et naturelle. "
        "Quand l'utilisateur demande la météo d'une ville, utilise l'outil get_weather. "
        "Villes disponibles : Paris, Lyon, Marseille, Bordeaux, Nice, Toulouse."
    ),
    tools=[get_weather],
    model="gpt-4o-mini",
)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def record_audio() -> np.ndarray:
    """Capture RECORD_SECONDS of mic audio and return a float32 numpy array."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print(f"  Enregistrement ({RECORD_SECONDS}s)… parlez maintenant.")
    frames = [
        stream.read(CHUNK, exception_on_overflow=False)
        for _ in range(int(SAMPLE_RATE / CHUNK * RECORD_SECONDS))
    ]
    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0


def play_audio(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Play raw int16 PCM bytes through the default output device."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=sample_rate,
        output=True,
    )
    stream.write(pcm_bytes)
    stream.stop_stream()
    stream.close()
    pa.terminate()


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set — check your .env file.")

    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

    print("=== Agent Vocal Français — Météo ===")
    print("Villes : Paris, Lyon, Marseille, Bordeaux, Nice, Toulouse")
    print("Appuyez sur Ctrl+C pour quitter.\n")

    while True:
        try:
            input("[ Appuyez sur Entrée pour parler… ]")
            audio_array = record_audio()

            print("  Traitement en cours…")
            result = await pipeline.run(AudioInput(buffer=audio_array))

            output_chunks: list[bytes] = []
            async for event in result.stream():
                if isinstance(event, VoiceStreamEventAudio):
                    output_chunks.append(event.data)

            if output_chunks:
                play_audio(b"".join(output_chunks))
            else:
                print("  (aucune réponse audio reçue)")

        except KeyboardInterrupt:
            print("\nAu revoir !")
            break


if __name__ == "__main__":
    asyncio.run(main())

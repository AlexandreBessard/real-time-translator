# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add OPENAI_API_KEY
```

## Running the scripts

```bash
# Real-time audio translator (WebSocket, gpt-realtime-translate)
python translator.py --source fr --target en
python translator.py --list-devices          # find audio device indices

# French voice agent — push-to-talk, STT→LLM→TTS via openai-agents SDK
python voice_agent_fr.py

# English teacher — live speech-to-speech via gpt-realtime-2 (OpenAI Realtime API)
python voice_agent_fr_realtime.py
# Env overrides:
#   INPUT_DEVICE="Blue Snowball"  — mic name substring or numeric index
#   VOICE=coral                   — OpenAI Realtime voice
#   ECHO_FROM_SPEAKERS=1          — mute mic during playback (open speakers)
```

## Serving the web front end

```bash
# Full stack (Emily + avatar) — use this:
python server.py          # serves web/ at http://localhost:8000 + POST /session
PORT=3000 python server.py

# Static-only (no Emily, no API key needed):
cd web && python3 -m http.server 8000
```

## Architecture

Three independent experiments sharing only the `.env` for the API key:

### `translator.py` — Real-time translation CLI
Direct WebSocket client to `wss://api.openai.com/v1/realtime/translations` (`gpt-realtime-translate`). Three daemon threads: capture (mic → WebSocket), receive (events → print + playback queue), playback (queue → speaker). Language codes are BCP-47 (e.g. `fr`, `en`, `es`).

### `voice_agent_fr.py` — Push-to-talk voice agent
Uses the `openai-agents` SDK (`VoicePipeline` + `SingleAgentVoiceWorkflow`). Architecture is chained STT → text agent (GPT-4o-mini + `get_weather` tool) → TTS. Records a fixed 5-second clip per button press; not streaming. French language throughout.

### `voice_agent_fr_realtime.py` — Live speech-to-speech agent (Emily)
Direct WebSocket to `wss://api.openai.com/v1/realtime` (`gpt-realtime-2`). `SESSION_CONFIG` at module level holds the full session schema (instructions, VAD config using `semantic_vad`, noise reduction, voice). Three daemon threads identical in shape to `translator.py`. Key complexity: half-duplex echo suppression (`ECHO_FROM_SPEAKERS` mode) mutes the mic while `_speaking` is set + a tail window after audio drains; full-duplex mode (headphones, default) leaves the mic always open so server VAD handles barge-in. Uses `resample_to_24k()` via linear interpolation to handle mics whose native rate ≠ 24 kHz (e.g. Blue Snowball at 44.1 kHz).

### `web/` — Talking avatar front end (static, no build step)
Three.js scene loaded via importmap from CDN. `main.js` loads a Ready Player Me `.glb` (or the bundled `brunette.glb`), indexes all morph targets across every mesh, and drives mouth blend shapes each frame from `LipSync.update()`. `lipsync.js` wraps a Web Audio `AnalyserNode` and derives two floats (`open`, `wide`) from RMS loudness and spectral balance. Supports mic, uploaded audio file, procedural test mode, and any external `MediaStream` via `lip.useStream(stream)` — the integration point for OpenAI Realtime WebRTC. PWA-ready (`manifest.webmanifest`, `sw.js`).

## Planned next step
Connect the web avatar to Emily's voice: add a tiny backend endpoint that mints an ephemeral Realtime token (so `OPENAI_API_KEY` never ships to the browser), then call `lip.useStream(remoteAudioStream)` with the WebRTC remote track.

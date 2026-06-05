# Emily — Talking Avatar (front end)

A responsive web app that renders a **Ready Player Me 3D avatar** and makes its
mouth move in real time from an audio source. Free, self-hosted, runs in
desktop **and** mobile browsers, no GPU. Installable as a PWA.

This is the **visual front end only**. Audio currently comes from a mock/test
source (test tone, mic, or an uploaded file). Wiring it to the OpenAI Realtime
voice is a later step — see *Next step* below.

## Run it (desktop)

From the repo root:

```bash
cd web
python3 -m http.server 8000
```

Open <http://localhost:8000>, tap **Start**, then try:

- **▶ Test (fake speech)** — preview lip sync with no input.
- **🎤 Microphone** — talk and watch the mouth follow you.
- **🎵 Audio file** — play any audio clip and lip-sync to it.

## Run it on your phone

The site must be reachable from the phone. Two options:

1. **Same Wi-Fi (quick look):** browse to `http://<your-computer-ip>:8000`.
   ⚠️ The **microphone is blocked over plain HTTP** on phones — *Test* and
   *Audio file* still work; mic needs HTTPS.
2. **HTTPS (full features):** expose it with a tunnel, e.g.
   `cloudflared tunnel --url http://localhost:8000` (or `ngrok http 8000`),
   then open the `https://…` URL on the phone. Mic works, and you can
   "Add to Home Screen" to install it.

## Use your own avatar

Tap **Change avatar**, create a free one at <https://readyplayer.me>, and paste
its `.glb` URL. The app appends `?morphTargets=ARKit,Oculus Visemes` so the face
has the blend shapes the lip sync drives.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Layout, importmap (three.js via CDN), controls |
| `main.js` | three.js scene, avatar load, morph targets, render loop |
| `lipsync.js` | Audio → mouth shape (`open`/`wide`), all sources |
| `style.css` | Responsive / mobile-safe styling |
| `manifest.webmanifest`, `sw.js`, `icon.svg` | PWA install support |

## Next step — connect Emily's real voice

The lip sync is source-agnostic. Once the browser holds Emily's audio as a
`MediaStream` (the standard way is OpenAI Realtime over **WebRTC**, which exposes
a remote audio track), drive the avatar with one call:

```js
lip.useStream(remoteAudioStream); // remoteAudioStream from the WebRTC track
```

That needs a tiny backend endpoint to mint an ephemeral Realtime token (so your
`OPENAI_API_KEY` never ships to the browser). Ask me to build that when ready.

## Going fully offline / self-hosted

`three.js` loads from a CDN for convenience. To remove that dependency, vendor
`three.module.js` and the `examples/jsm/` addons into `web/vendor/` and point the
importmap in `index.html` at the local copies.

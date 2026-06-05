// emily.js — WebRTC connection to the Emily voice agent via OpenAI Realtime.
//
// Flow:
//   1. POST /session  → backend mints an ephemeral token (OPENAI_API_KEY stays server-side)
//   2. RTCPeerConnection + mic track + data channel
//   3. Wait for ICE gathering to complete, then POST the full SDP offer
//   4. Set remote description from SDP answer  ← connect() resolves here
//   5. Remote audio track → <audio> element (playback) + LipSync.useStream() (mouth)
//
// Callbacks (set before calling connect()):
//   emily.onTranscript  = (speaker, text) => {}   // 'user' | 'emily'
//   emily.onDisconnect  = () => {}                // remote side closed

export class EmilyRealtime {
  constructor(lip) {
    this.lip = lip;
    this.onTranscript = null;
    this.onDisconnect = null;
    this._lastEmilyText = "";   // dedupe consecutive identical transcript events

    this._pc = null;
    this._dc = null;
    this._audioEl = null;
  }

  get connected() {
    return this._pc !== null && this._pc.connectionState !== "closed";
  }

  async connect() {
    // ── 1. Mint ephemeral token ────────────────────────────────
    const res = await fetch("/session", { method: "POST" });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(`/session ${res.status}: ${msg}`);
    }
    const { value: token } = await res.json();
    console.log("[Emily] session token received");

    // ── 2. Peer connection ─────────────────────────────────────
    this._pc = new RTCPeerConnection();

    this._pc.onconnectionstatechange = () => {
      const s = this._pc?.connectionState;
      console.log("[Emily] connectionState →", s);
      if (s === "disconnected" || s === "failed" || s === "closed") {
        this.onDisconnect?.();
      }
    };
    this._pc.oniceconnectionstatechange = () =>
      console.log("[Emily] iceConnectionState →", this._pc?.iceConnectionState);

    // ── 3. Remote audio track → playback + lip sync ────────────
    this._pc.ontrack = (e) => {
      if (e.track.kind !== "audio") return;
      console.log("[Emily] remote audio track received");
      const stream = new MediaStream([e.track]);

      if (!this._audioEl) {
        this._audioEl = new Audio();
        this._audioEl.autoplay = true;
      }
      this._audioEl.srcObject = stream;

      // owned:false → LipSync.stop() won't call track.stop() on a WebRTC track
      this.lip.useStream(stream, { owned: false });
    };

    // ── 4. Microphone → outbound track ────────────────────────
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micStream.getTracks().forEach((t) => this._pc.addTrack(t));
    console.log("[Emily] mic track added");

    // ── 5. Data channel for transcript events ─────────────────
    this._dc = this._pc.createDataChannel("oai-events");
    this._dc.addEventListener("message", (e) => {
      try { this._onEvent(JSON.parse(e.data)); } catch {}
    });
    this._dc.addEventListener("open",  () => console.log("[Emily] data channel open"));
    this._dc.addEventListener("close", () => console.log("[Emily] data channel closed"));

    // ── 6. Create offer and wait for ICE gathering ────────────
    // ICE candidates are gathered asynchronously after setLocalDescription.
    // We must wait for iceGatheringState==='complete' before sending the SDP —
    // otherwise the offer has no candidates and OpenAI can't route audio back.
    const offer = await this._pc.createOffer();
    await this._pc.setLocalDescription(offer);
    console.log("[Emily] waiting for ICE gathering…");

    await new Promise((resolve) => {
      if (this._pc.iceGatheringState === "complete") { resolve(); return; }
      const onGather = () => {
        if (this._pc?.iceGatheringState === "complete") {
          this._pc.removeEventListener("icegatheringstatechange", onGather);
          resolve();
        }
      };
      this._pc.addEventListener("icegatheringstatechange", onGather);
    });
    console.log("[Emily] ICE gathering complete");

    // ── 7. Exchange SDP with OpenAI ────────────────────────────
    // Use this._pc.localDescription.sdp (not offer.sdp) — it contains the
    // gathered ICE candidates that were appended after setLocalDescription.
    const sdpRes = await fetch(
      "https://api.openai.com/v1/realtime/calls",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/sdp",
        },
        body: this._pc.localDescription.sdp,
      }
    );
    if (!sdpRes.ok) {
      throw new Error(
        `OpenAI WebRTC ${sdpRes.status}: ${await sdpRes.text().catch(() => "")}`
      );
    }

    const answerSdp = await sdpRes.text();
    console.log("[Emily] SDP answer received, setting remote description");
    await this._pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
    // connect() resolves here; audio will start flowing once ICE/DTLS complete
  }

  disconnect() {
    this.lip.stop();
    if (this._audioEl) {
      this._audioEl.srcObject = null;
      this._audioEl = null;
    }
    if (this._dc) {
      try { this._dc.close(); } catch {}
      this._dc = null;
    }
    if (this._pc) {
      try { this._pc.close(); } catch {}
      this._pc = null;
    }
  }

  // ── Data channel event handler ─────────────────────────────

  _onEvent(event) {
    const t = event.type;

    // User's speech → transcript
    if (t === "conversation.item.input_audio_transcription.completed") {
      const text = event.transcript?.trim();
      if (text) this.onTranscript?.("user", text);
    }

    // Emily's spoken response → transcript. The GA Realtime API renamed this
    // event with an "output_" prefix; accept both so beta and GA both work.
    if (t === "response.output_audio_transcript.done" ||
        t === "response.audio_transcript.done") {
      const text = event.transcript?.trim();
      // Some responses surface the transcript via more than one event — only
      // act on a genuinely new line so we don't classify/show it twice.
      if (text && text !== this._lastEmilyText) {
        console.log(`[emily] transcript via ${t}`);
        this._lastEmilyText = text;
        this.onTranscript?.("emily", text);
      } else if (text) {
        console.log(`[emily] duplicate transcript via ${t} — ignored`);
      }
    }
  }
}

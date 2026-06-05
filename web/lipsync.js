// lipsync.js — turns an audio source into a simple mouth shape, in real time.
//
// It taps any Web Audio source with an AnalyserNode and, every frame, derives:
//   - open : how far the mouth opens (overall loudness)             0..1
//   - wide : spread vs. round (low- vs. high-frequency energy)      0..1
//
// This is amplitude/spectral lip "flap" — not true phoneme visemes — but it
// reads convincingly for conversation and runs entirely client-side, free,
// on desktop and mobile. When you later feed it Emily's WebRTC audio track,
// nothing here changes: just call useStream(remoteStream).

export class LipSync {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.timeData = null;
    this.freqData = null;

    // Smoothed outputs (attack fast, release slower → natural-looking).
    this.open = 0;
    this.wide = 0;
    this.energy = 0;       // slow loudness envelope → "arousal" for expression

    this._active = null;   // current source node we may need to stop
    this._test = false;    // procedural "fake speech" mode
    this._t = 0;
  }

  _ensureCtx() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.6;
      this.timeData = new Uint8Array(this.analyser.fftSize);
      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
    }
    // Browsers start the context suspended until a user gesture.
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  // Wake the audio context from a user gesture (the Start button).
  async unlock() {
    this._ensureCtx();
  }

  stop() {
    this._test = false;
    if (this._active) {
      try { this._active.stop?.(); } catch {}
      try { this._active.disconnect?.(); } catch {}
      try { this._active.mediaStream?.getTracks().forEach((t) => t.stop()); } catch {}
      this._active = null;
    }
  }

  // ── Sources ────────────────────────────────────────────────

  // Live microphone — best way to eyeball the lip sync by talking.
  async useMicrophone() {
    this.stop();
    const ctx = this._ensureCtx();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = ctx.createMediaStreamSource(stream);
    src.mediaStream = stream;            // keep a handle so stop() can end it
    src.connect(this.analyser);          // analyse only — don't echo mic to speakers
    this._active = src;
  }

  // Any external MediaStream (e.g. the OpenAI Realtime WebRTC track).
  // Pass { owned: false } for streams whose track lifecycle is managed externally
  // (e.g. a WebRTC peer connection) so stop() won't call track.stop() on them.
  async useStream(stream, { owned = true } = {}) {
    this.stop();
    const ctx = this._ensureCtx();
    const src = ctx.createMediaStreamSource(stream);
    if (owned) src.mediaStream = stream;
    src.connect(this.analyser);
    this._active = src;
  }

  // Decode and play an uploaded audio file; we both hear it and lip-sync to it.
  async useFile(file) {
    this.stop();
    const ctx = this._ensureCtx();
    const buf = await file.arrayBuffer();
    const audioBuf = await ctx.decodeAudioData(buf);
    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(this.analyser);
    this.analyser.connect(ctx.destination);  // route to speakers so we hear it
    src.start();
    this._active = src;
    return new Promise((resolve) => { src.onended = resolve; });
  }

  // Procedural "fake speech" — preview lip sync with no mic or file at all.
  useTest() {
    this.stop();
    this._ensureCtx();
    this._test = true;
    this._t = 0;
  }

  // ── Per-frame update ───────────────────────────────────────

  update(dt) {
    let targetOpen = 0;
    let targetWide = 0.3;

    if (this._test) {
      // Stack a few sines so the mouth opens in irregular, speech-like bursts,
      // with pauses between "words".
      this._t += dt;
      const t = this._t;
      const syllable = Math.max(0, Math.sin(t * 9) * 0.6 + Math.sin(t * 5.3) * 0.4);
      const gate = (Math.sin(t * 1.7) + 1) * 0.5 > 0.35 ? 1 : 0; // breathy pauses
      targetOpen = syllable * gate;
      targetWide = (Math.sin(t * 2.1) + 1) * 0.5;
    } else if (this.analyser) {
      // Loudness (RMS of the waveform) → how open the mouth is.
      this.analyser.getByteTimeDomainData(this.timeData);
      let sum = 0;
      for (let i = 0; i < this.timeData.length; i++) {
        const v = (this.timeData[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / this.timeData.length);
      targetOpen = Math.min(1, rms * 3.2); // scale quiet speech up a bit

      // Spectral balance → spread (high freq = "ee") vs round (low freq = "oo").
      this.analyser.getByteFrequencyData(this.freqData);
      const n = this.freqData.length;
      let low = 0, high = 0;
      for (let i = 0; i < n; i++) {
        if (i < n * 0.25) low += this.freqData[i];
        else high += this.freqData[i];
      }
      targetWide = high / (low + high + 1);
    }

    // Asymmetric smoothing: snap open, ease closed → no robotic flapping.
    const aOpen = targetOpen > this.open ? 0.5 : 0.2;
    this.open += (targetOpen - this.open) * aOpen;
    this.wide += (targetWide - this.wide) * 0.15;

    // Much slower envelope of loudness: a sense of how animated the speech is
    // over the last ~second, not this instant. Drives expressive liveliness
    // (brow lifts, eye engagement) without flickering frame to frame.
    this.energy += (this.open - this.energy) * (this.open > this.energy ? 0.04 : 0.02);

    return { open: this.open, wide: this.wide, energy: this.energy };
  }
}

// expression.js — emotional life on top of lip-sync.
//
// LipSync owns the mouth-open shape. This module owns everything else that
// makes a face read as a person rather than a puppet: brows, eyes (widen /
// squint), cheeks, smile / frown, gaze saccades, blinking, breathing and head
// micro-motion.
//
// It is driven two ways at once:
//   • prosody  — the per-frame {open, energy} from LipSync gives liveliness:
//                brows lift on vocal emphasis, eyes engage when she's animated.
//   • emotion  — setEmotion('happy' | 'curious' | …), chosen from transcript
//                sentiment, sets a target facial pose she eases into and then
//                relaxes out of after a few seconds.
//
// update() returns plain numbers (morph-name → 0..1, plus head-rotation
// offsets) so it stays renderer-agnostic. main.js applies them and silently
// ignores any morph the loaded model doesn't have — so the same presets work
// on ARKit heads and Ready Player Me avatars alike.

// Facial poses as ARKit blend-shape weights. Slight asymmetry (left ≠ right)
// makes the face read as a real person rather than a mirrored puppet.
const PRESETS = {
  neutral:   { mouthSmileLeft: 0.10, mouthSmileRight: 0.10, browInnerUp: 0.05 },
  happy:     { mouthSmileLeft: 0.58, mouthSmileRight: 0.54,
               cheekSquintLeft: 0.45, cheekSquintRight: 0.42,
               eyeSquintLeft: 0.22, eyeSquintRight: 0.20,
               browInnerUp: 0.12 },
  curious:   { browInnerUp: 0.48, browOuterUpLeft: 0.42, browOuterUpRight: 0.38,
               eyeWideLeft: 0.20, eyeWideRight: 0.18,
               mouthSmileLeft: 0.22, mouthSmileRight: 0.20 },
  surprised: { browInnerUp: 0.72, browOuterUpLeft: 0.68, browOuterUpRight: 0.65,
               eyeWideLeft: 0.58, eyeWideRight: 0.55, jawOpen: 0.14 },
  concerned: { browInnerUp: 0.55, browDownLeft: 0.20, browDownRight: 0.18,
               mouthFrownLeft: 0.22, mouthFrownRight: 0.20,
               eyeSquintLeft: 0.08, eyeSquintRight: 0.08 },
  thinking:  { browDownLeft: 0.30, browDownRight: 0.25,
               eyeSquintLeft: 0.22, eyeSquintRight: 0.18,
               mouthPucker: 0.16, mouthLeft: 0.06 },
  // Attentive, open face while listening — brows slightly raised, gentle smile.
  listening: { browInnerUp: 0.22, browOuterUpLeft: 0.16, browOuterUpRight: 0.14,
               eyeWideLeft: 0.08, eyeWideRight: 0.07,
               mouthSmileLeft: 0.15, mouthSmileRight: 0.15 },
};

// How long each emotion holds before relaxing back to neutral.
const DEFAULT_HOLD = {
  neutral:   0,
  happy:     4.5,
  curious:   5.0,
  surprised: 2.5,
  concerned: 4.0,
  thinking:  3.5,
  listening: 5.0,
};

// Lightweight sentiment → emotion classifier.
const SURPRISE  = /\b(wow|whoa|oh|really|no way|incredible|amazing|unbelievable)\b/i;
const POSITIVE  = /\b(great|good job|well done|perfect|excellent|exactly|awesome|wonderful|fantastic|brilliant|nicely|yay|correct|bravo|love it|that's right|well said)\b/i;
const CONCERN   = /\b(sorry|not quite|careful|almost|don't worry|unfortunately|mistake|hmm,? no|that's not quite)\b/i;
const THINK     = /\b(let me think|let's see|i wonder|hmm|well,|interesting|actually|let me)\b/i;
const ENCOURAGE = /\b(keep going|you can|try again|good try|nice try|almost there|getting better)\b/i;

export function classifyEmotion(text) {
  const t = (text || "").trim();
  if (!t) return "neutral";
  if (SURPRISE.test(t))  return "surprised";
  if (POSITIVE.test(t))  return "happy";
  if (ENCOURAGE.test(t)) return "happy";
  if (CONCERN.test(t))   return "concerned";
  if (THINK.test(t))     return "thinking";
  if (t.endsWith("?"))   return "curious";
  return "neutral";
}

export class Expression {
  constructor() {
    this.weights = {};
    this.target  = { ...PRESETS.neutral };
    this.mood    = "neutral";
    this.holdT   = 0;

    // Blink scheduling.
    this.nextBlink  = 1 + Math.random() * 3;
    this.blinkT     = -1;
    this.blinkQueue = 0;

    // Gaze saccades.
    this.gaze       = { x: 0, y: 0 };
    this.gazeTarget = { x: 0, y: 0 };
    this.nextSaccade = 0.6;

    // One-shot nod: a forward-back head dip to signal acknowledgment.
    this._nod = { active: false, t: 0, amp: 0, dur: 0.40 };

    // Smooth head tilt offset (rz): curious tilts right, thinking tilts left.
    this._tilt = 0;

    // Fixed random phases so summed sines read as organic noise.
    this.ph = Array.from({ length: 6 }, () => Math.random() * Math.PI * 2);
  }

  setEmotion(name, hold) {
    if (!PRESETS[name]) name = "neutral";
    this.mood   = name;
    this.target = { ...PRESETS[name] };
    this.holdT  = hold !== undefined ? hold : (DEFAULT_HOLD[name] ?? 3.5);
  }

  // Play a one-shot forward nod (head.rx bump that fades out over `dur` s).
  triggerNod(amplitude = 0.07, dur = 0.40) {
    this._nod = { active: true, t: 0, amp: amplitude, dur };
  }

  update(dt, elapsed, { open = 0, energy = 0, speaking = false } = {}) {
    // Relax to neutral once the hold window expires.
    this.holdT -= dt;
    if (this.holdT <= 0 && this.mood !== "neutral") {
      this.mood   = "neutral";
      this.target = { ...PRESETS.neutral };
    }

    // Ease every channel toward the target pose.
    const keys = new Set([...Object.keys(this.weights), ...Object.keys(this.target)]);
    const k = Math.min(1, dt * 4);
    for (const key of keys) {
      const cur  = this.weights[key] || 0;
      const next = cur + ((this.target[key] || 0) - cur) * k;
      if (next < 0.001) delete this.weights[key];
      else this.weights[key] = next;
    }

    const morphs = { ...this.weights };

    // ── Prosody: liveliness while speaking ────────────────────────────────────
    if (speaking) {
      const emph = Math.min(1, open * 1.2) * (0.4 + energy * 0.6);
      addMax(morphs, "browInnerUp",       emph * 0.28);
      addMax(morphs, "browOuterUpLeft",   emph * 0.20);
      addMax(morphs, "browOuterUpRight",  emph * 0.16);  // slight asymmetry = personality
      addMax(morphs, "eyeWideLeft",       energy * 0.14);
      addMax(morphs, "eyeWideRight",      energy * 0.12);
      addMax(morphs, "mouthStretchLeft",  emph * 0.12);  // lip corners engage on emphasis
      addMax(morphs, "mouthStretchRight", emph * 0.10);
      addMax(morphs, "cheekSquintLeft",   energy * 0.16);
      addMax(morphs, "cheekSquintRight",  energy * 0.14);
    }

    // ── Blink — interval varies by mood ───────────────────────────────────────
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT < 0) {
      this.blinkT = 0;
      const [lo, hi] = ({
        surprised: [0.8,  1.5],
        thinking:  [3.5,  6.0],
        concerned: [1.5,  3.0],
      })[this.mood] || [2.5, 5.0];
      this.nextBlink  = lo + Math.random() * (hi - lo);
      this.blinkQueue = Math.random() < 0.2 ? 1 : 0;   // ~20% double-blink
    }
    if (this.blinkT >= 0) {
      this.blinkT += dt;
      const p = this.blinkT / 0.14;
      const v = p < 1 ? Math.sin(p * Math.PI) : 0;
      morphs.eyeBlinkLeft  = v;
      morphs.eyeBlinkRight = v;
      if (p >= 1) {
        this.blinkT = -1;
        if (this.blinkQueue > 0) { this.blinkQueue--; this.nextBlink = 0.12; }
      }
    }

    // ── Gaze saccades — biased toward center for natural eye contact ──────────
    this.nextSaccade -= dt;
    if (this.nextSaccade <= 0) {
      const r = Math.random();
      if (this.mood === "thinking" && r < 0.55) {
        // Up-and-away reads as deep thought.
        this.gazeTarget = { x: (Math.random() - 0.5) * 1.2, y: 0.5 + Math.random() * 0.4 };
        this.nextSaccade = 0.7 + Math.random() * 1.2;
      } else if (r < 0.55) {
        // Eye contact — near center, held longer.
        this.gazeTarget = { x: (Math.random() - 0.5) * 0.22, y: (Math.random() - 0.5) * 0.14 };
        this.nextSaccade = 1.0 + Math.random() * 2.5;
      } else if (r < 0.82) {
        // Small drift.
        this.gazeTarget = { x: (Math.random() - 0.5) * 0.55, y: (Math.random() - 0.5) * 0.30 };
        this.nextSaccade = 0.5 + Math.random() * 1.5;
      } else {
        // Rare larger glance away.
        this.gazeTarget = { x: (Math.random() - 0.5) * 1.2, y: (Math.random() - 0.5) * 0.7 };
        this.nextSaccade = 0.25 + Math.random() * 0.5;
      }
    }
    const gk = Math.min(1, dt * 18);
    this.gaze.x += (this.gazeTarget.x - this.gaze.x) * gk;
    this.gaze.y += (this.gazeTarget.y - this.gaze.y) * gk;
    const gx = this.gaze.x, gy = this.gaze.y;
    if (gx < 0) { addMax(morphs, "eyeLookOutLeft",  -gx * 0.3); addMax(morphs, "eyeLookInRight",  -gx * 0.3); }
    else        { addMax(morphs, "eyeLookInLeft",    gx * 0.3);  addMax(morphs, "eyeLookOutRight",  gx * 0.3); }
    if (gy > 0) { addMax(morphs, "eyeLookUpLeft",    gy * 0.3);  addMax(morphs, "eyeLookUpRight",   gy * 0.3); }
    else        { addMax(morphs, "eyeLookDownLeft",  -gy * 0.3); addMax(morphs, "eyeLookDownRight", -gy * 0.3); }

    // ── Head: breathing + organic micro-motion + gaze follow + tilt ──────────
    const ph = this.ph;
    const breath = Math.sin(elapsed * 0.7 + ph[0]) * 0.012;
    const noiseX = (Math.sin(elapsed * 0.53 + ph[1]) + Math.sin(elapsed * 0.31 + ph[2])) * 0.012;
    const noiseY = (Math.sin(elapsed * 0.47 + ph[3]) + Math.sin(elapsed * 0.27 + ph[4])) * 0.015;

    // Head tilt: curious tilts slightly right, thinking slightly left.
    const tiltTarget = this.mood === "curious" ? 0.045 : (this.mood === "thinking" ? -0.025 : 0);
    this._tilt += (tiltTarget - this._tilt) * Math.min(1, dt * 1.2);

    const head = {
      rx: breath + noiseX - gy * 0.05,
      ry: noiseY + gx * 0.06,
      rz: Math.sin(elapsed * 0.23 + ph[5]) * 0.008 + this._tilt,
    };

    // One-shot nod: smooth half-sine bump on rx.
    if (this._nod.active) {
      this._nod.t += dt;
      const p = this._nod.t / this._nod.dur;
      if (p >= 1) {
        this._nod.active = false;
      } else {
        head.rx += Math.sin(p * Math.PI) * this._nod.amp;
      }
    }

    return { morphs, head };
  }
}

function addMax(obj, key, v) {
  if (v > (obj[key] || 0)) obj[key] = v;
}

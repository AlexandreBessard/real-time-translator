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

// Facial poses as ARKit blend-shape weights. Names a given model lacks are
// skipped by the caller. "neutral" is a gentle, pleasant resting face — a
// teacher should never look dead-eyed.
const PRESETS = {
  neutral:   { mouthSmileLeft: 0.10, mouthSmileRight: 0.10, browInnerUp: 0.05 },
  happy:     { mouthSmileLeft: 0.55, mouthSmileRight: 0.55, cheekSquintLeft: 0.45,
               cheekSquintRight: 0.45, eyeSquintLeft: 0.22, eyeSquintRight: 0.22,
               browInnerUp: 0.10 },
  curious:   { browInnerUp: 0.45, browOuterUpLeft: 0.40, browOuterUpRight: 0.40,
               eyeWideLeft: 0.18, eyeWideRight: 0.18, mouthSmileLeft: 0.20,
               mouthSmileRight: 0.20 },
  surprised: { browInnerUp: 0.70, browOuterUpLeft: 0.65, browOuterUpRight: 0.65,
               eyeWideLeft: 0.55, eyeWideRight: 0.55, jawOpen: 0.12 },
  concerned: { browInnerUp: 0.55, browDownLeft: 0.18, browDownRight: 0.18,
               mouthFrownLeft: 0.22, mouthFrownRight: 0.22 },
  thinking:  { browDownLeft: 0.28, browDownRight: 0.28, eyeSquintLeft: 0.20,
               eyeSquintRight: 0.20, mouthPucker: 0.15 },
};

// Lightweight sentiment → emotion. Coarse on purpose: it only needs to pick a
// plausible mood for a spoken line, and a wrong guess relaxes away in seconds.
const SURPRISE = /\b(wow|whoa|oh|really|no way|incredible|amazing)\b/i;
const POSITIVE = /\b(great|good job|well done|perfect|excellent|exactly|awesome|wonderful|fantastic|brilliant|nicely|yay|correct)\b/i;
const CONCERN  = /\b(sorry|not quite|careful|almost|don't worry|unfortunately|mistake|hmm,? no)\b/i;
const THINK    = /\b(let me think|let's see|i wonder|hmm|well,)\b/i;

export function classifyEmotion(text) {
  const t = (text || "").trim();
  if (!t) return "neutral";
  if (SURPRISE.test(t)) return "surprised";
  if (POSITIVE.test(t)) return "happy";
  if (CONCERN.test(t))  return "concerned";
  if (THINK.test(t))    return "thinking";
  if (t.endsWith("?"))  return "curious";
  return "neutral";
}

export class Expression {
  constructor() {
    this.weights = {};                  // current eased pose
    this.target = { ...PRESETS.neutral };
    this.mood = "neutral";
    this.holdT = 0;                     // seconds left before relaxing to neutral

    // Blink scheduling.
    this.nextBlink = 1 + Math.random() * 3;
    this.blinkT = -1;
    this.blinkQueue = 0;

    // Gaze: a target the eyes dart toward, eased fast (saccades are ballistic).
    this.gaze = { x: 0, y: 0 };
    this.gazeTarget = { x: 0, y: 0 };
    this.nextSaccade = 0.6;

    // Fixed random phases so summed sines read as organic noise, not a clean
    // oscillation. (Math.random is fine in the browser.)
    this.ph = Array.from({ length: 6 }, () => Math.random() * Math.PI * 2);
  }

  // Set a target emotional pose she eases into, holding it for `hold` seconds
  // before relaxing back to neutral.
  setEmotion(name, hold = 3.5) {
    if (!PRESETS[name]) name = "neutral";
    this.mood = name;
    this.target = { ...PRESETS[name] };
    this.holdT = hold;
  }

  // Returns { morphs, head } for this frame. `input` carries prosody from
  // LipSync: open (0..1 instantaneous), energy (0..1 slow), speaking (bool).
  update(dt, elapsed, { open = 0, energy = 0, speaking = false } = {}) {
    // Relax to the resting face once the hold window expires.
    this.holdT -= dt;
    if (this.holdT <= 0 && this.mood !== "neutral") {
      this.mood = "neutral";
      this.target = { ...PRESETS.neutral };
    }

    // Ease every channel toward the target pose. Union of keys so channels the
    // new pose drops also ease back to 0.
    const keys = new Set([...Object.keys(this.weights), ...Object.keys(this.target)]);
    const k = Math.min(1, dt * 4);
    for (const key of keys) {
      const cur = this.weights[key] || 0;
      const next = cur + ((this.target[key] || 0) - cur) * k;
      if (next < 0.001) delete this.weights[key];
      else this.weights[key] = next;
    }

    const morphs = { ...this.weights };

    // ── Prosody: liveliness while speaking ────────────────────
    // Brows lift on loud/emphatic moments; eyes engage with sustained energy.
    if (speaking) {
      const emph = Math.min(1, open * 1.2) * (0.4 + energy * 0.6);
      addMax(morphs, "browInnerUp", emph * 0.25);
      addMax(morphs, "browOuterUpLeft", emph * 0.18);
      addMax(morphs, "browOuterUpRight", emph * 0.18);
      addMax(morphs, "eyeWideLeft", energy * 0.12);
      addMax(morphs, "eyeWideRight", energy * 0.12);
    }

    // ── Blink ─────────────────────────────────────────────────
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT < 0) {
      this.blinkT = 0;
      this.nextBlink = 2 + Math.random() * 4;
      this.blinkQueue = Math.random() < 0.2 ? 1 : 0;   // ~20% double blink
    }
    if (this.blinkT >= 0) {
      this.blinkT += dt;
      const p = this.blinkT / 0.14;
      const v = p < 1 ? Math.sin(p * Math.PI) : 0;
      morphs.eyeBlinkLeft = v;
      morphs.eyeBlinkRight = v;
      if (p >= 1) {
        this.blinkT = -1;
        if (this.blinkQueue > 0) { this.blinkQueue--; this.nextBlink = 0.12; }
      }
    }

    // ── Gaze: micro-saccades around the camera ────────────────
    this.nextSaccade -= dt;
    if (this.nextSaccade <= 0) {
      const r = Math.random();
      if (this.mood === "thinking" && r < 0.6) {
        // Up-and-away reads as "thinking".
        this.gazeTarget = { x: (Math.random() - 0.5) * 1.2, y: 0.5 + Math.random() * 0.4 };
        this.nextSaccade = 0.7 + Math.random() * 1.2;
      } else if (r < 0.15) {
        // Occasional larger glance away, then back.
        this.gazeTarget = { x: (Math.random() - 0.5) * 1.4, y: (Math.random() - 0.5) * 0.8 };
        this.nextSaccade = 0.3 + Math.random() * 0.6;
      } else {
        // Mostly small darts near the viewer.
        this.gazeTarget = { x: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.3 };
        this.nextSaccade = 0.5 + Math.random() * 2;
      }
    }
    const gk = Math.min(1, dt * 18);   // fast — saccades are near-instant
    this.gaze.x += (this.gazeTarget.x - this.gaze.x) * gk;
    this.gaze.y += (this.gazeTarget.y - this.gaze.y) * gk;
    const gx = this.gaze.x, gy = this.gaze.y;
    if (gx < 0) { addMax(morphs, "eyeLookOutLeft", -gx * 0.3); addMax(morphs, "eyeLookInRight", -gx * 0.3); }
    else        { addMax(morphs, "eyeLookInLeft", gx * 0.3);   addMax(morphs, "eyeLookOutRight", gx * 0.3); }
    if (gy > 0) { addMax(morphs, "eyeLookUpLeft", gy * 0.3);   addMax(morphs, "eyeLookUpRight", gy * 0.3); }
    else        { addMax(morphs, "eyeLookDownLeft", -gy * 0.3); addMax(morphs, "eyeLookDownRight", -gy * 0.3); }

    // ── Head: breathing + organic micro-motion + slight gaze follow ──
    const ph = this.ph;
    const breath = Math.sin(elapsed * 0.7 + ph[0]) * 0.012;
    const noiseX = (Math.sin(elapsed * 0.53 + ph[1]) + Math.sin(elapsed * 0.31 + ph[2])) * 0.012;
    const noiseY = (Math.sin(elapsed * 0.47 + ph[3]) + Math.sin(elapsed * 0.27 + ph[4])) * 0.015;
    const head = {
      rx: breath + noiseX - gy * 0.05,   // glancing down dips the chin a little
      ry: noiseY + gx * 0.06,            // head turns slightly with the eyes
      rz: Math.sin(elapsed * 0.23 + ph[5]) * 0.01,
    };

    return { morphs, head };
  }
}

function addMax(obj, key, v) {
  if (v > (obj[key] || 0)) obj[key] = v;
}

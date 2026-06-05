// main.js — renders a Ready Player Me 3D avatar and drives its mouth from audio.
//
// Scope: visual front end only. Audio comes from a mock/test source (test tone,
// microphone, or an uploaded file). Swapping in OpenAI Realtime later is a
// one-liner: lip.useStream(remoteAudioStream). See web/README.md.

import * as THREE from "three";
import { EmilyRealtime } from "./emily.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LipSync } from "./lipsync.js";
import { Expression, classifyEmotion } from "./expression.js";

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/";

// Default: a realistic scanned head from the three.js examples — a reliable
// host that ships ARKit facial blend shapes, so the face moves out of the box.
// Paste a Ready Player Me .glb in the UI for a full stylized avatar; RPM URLs
// get the morph-target query params appended automatically.
const DEFAULT_AVATAR = "./brunette.glb";
const RPM_MORPH_PARAMS = "morphTargets=ARKit,Oculus Visemes&textureAtlas=1024&lod=0";

const canvas = document.getElementById("avatar-canvas");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-btn");
const controls = document.getElementById("controls");
const repeatCard = document.getElementById("repeat-card");
const repeatLabel = document.getElementById("repeat-label");
const repeatWord = document.getElementById("repeat-word");
const repeatText = document.getElementById("repeat-text");

const lip = new LipSync();
const face = new Expression();
// Dev hooks: try these live from the console.
//   setEmotion("surprised")          — preview an expression
//   showRepeat("I went to the park") — preview the repeat-after-me card
window.setEmotion = (name) => face.setEmotion(name, 6);
window.showRepeat = (sentence) => showRepeat(sentence);

// ── three.js scene ───────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 100);

// Soft studio lighting so skin reads naturally.
scene.add(new THREE.HemisphereLight(0xffffff, 0x444466, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(1, 2, 2.5);
scene.add(key);
const fill = new THREE.DirectionalLight(0x88aaff, 0.5);
fill.position.set(-2, 1, 1);
scene.add(fill);

// A single GLTFLoader, wired with the decoders the sample models need:
//   • KTX2  — facecap.glb ships Basis-compressed (KTX2) textures
//   • Draco — common mesh compression for RPM/other .glb files
//   • Meshopt — geometry compression RPM also uses
// Without these the loader throws ("setKTX2Loader must be called…") and nothing
// renders. Transcoders are pulled from the same CDN/version as three itself.
const gltfLoader = new GLTFLoader();
const ktx2Loader = new KTX2Loader()
  .setTranscoderPath(THREE_CDN + "libs/basis/")
  .detectSupport(renderer);
const dracoLoader = new DRACOLoader().setDecoderPath(THREE_CDN + "libs/draco/");
gltfLoader.setKTX2Loader(ktx2Loader);
gltfLoader.setDRACOLoader(dracoLoader);
gltfLoader.setMeshoptDecoder(MeshoptDecoder);

const orbit = new OrbitControls(camera, canvas);
orbit.enablePan = false;
orbit.enableZoom = false;
orbit.minPolarAngle = Math.PI * 0.42;
orbit.maxPolarAngle = Math.PI * 0.55;
orbit.minAzimuthAngle = -0.5;
orbit.maxAzimuthAngle = 0.5;
orbit.rotateSpeed = 0.4;

// ── Avatar state ─────────────────────────────────────────────
let avatar = null;          // the loaded model root
let headBone = null;        // for subtle idle motion
const morphTargets = {};    // name → [{ mesh, index }, ...]

// Blend-shape names per "channel". We list candidates from both the Oculus
// viseme set (Ready Player Me) and the ARKit set (facecap & RPM) so the same
// drive code animates whichever the loaded model actually has — missing names
// are simply skipped.
const JAW   = ["jawOpen"];                                   // both sets
const AA    = ["viseme_aa"];                                 // RPM open vowel
const WIDE  = ["viseme_I", "mouthSmileLeft", "mouthSmileRight"];
const ROUND = ["viseme_U", "mouthFunnel", "mouthPucker"];

function indexMorphs(model) {
  Object.keys(morphTargets).forEach((k) => delete morphTargets[k]);
  model.traverse((o) => {
    if (!o.isMesh || !o.morphTargetDictionary) return;
    for (const [name, idx] of Object.entries(o.morphTargetDictionary)) {
      (morphTargets[name] ||= []).push({ mesh: o, index: idx });
    }
  });
}

// Raise a set of candidate blend shapes to `value` in an influence buffer,
// keeping the strongest contribution per channel (max, not sum) so an
// emotional smile and a viseme spread don't stack past 1.0 and fight.
function mergeMax(infl, names, value) {
  for (const name of names) if (value > (infl[name] || 0)) infl[name] = value;
}

// Write the per-frame influence buffer to the model. Every morph the model has
// is set explicitly, so any channel absent from the buffer relaxes to 0 — no
// stuck expressions, no leftover visemes.
function applyMorphs(infl) {
  for (const name in morphTargets) {
    const v = infl[name] || 0;
    for (const { mesh, index } of morphTargets[name]) mesh.morphTargetInfluences[index] = v;
  }
}

async function loadAvatar(url) {
  statusEl.textContent = "Loading avatar…";
  const isRPM = url.includes("readyplayer.me");
  const full = isRPM ? url + (url.includes("?") ? "&" : "?") + RPM_MORPH_PARAMS : url;
  try {
    const gltf = await new Promise((res, rej) => gltfLoader.load(full, res, undefined, rej));
    if (avatar) scene.remove(avatar);
    avatar = gltf.scene;
    scene.add(avatar);
    indexMorphs(avatar);
    headBone = avatar.getObjectByName("Head") || null;
    frameOnFace();
    statusEl.textContent = "Ready";
    return true;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Couldn’t load that avatar — check the .glb URL";
    return false;
  }
}

// Frame the camera on the face as a portrait. Works across very different
// models — a full-body Ready Player Me avatar, a clean head, or a raw head
// *scan* like facecap.glb whose mesh extends down into the neck/shoulders so
// its bounding box is no guide to where the face actually is.
//
// Strategy: anchor on the small "feature" meshes (eyes, teeth) when present.
// They sit exactly on the face, so their combined box is a far more reliable
// target than the whole model. If the feature cluster lands in the MIDDLE of
// the model (a head scan), we frame it directly; if it's near the TOP (a body
// standing below the head) or there are no feature meshes, we fall back to the
// size-based heuristic that frames the upper slice / whole head.
function frameOnFace() {
  const box = new THREE.Box3().setFromObject(avatar);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  // Best case: use the Head bone world position directly — works for any
  // full-body avatar (RPM, TalkingHead, etc.) without geometry heuristics.
  if (headBone) {
    headBone.updateWorldMatrix(true, false);
    const headPos = new THREE.Vector3().setFromMatrixPosition(headBone.matrixWorld);
    const viewH = size.y * 0.22;
    const dist = (viewH / 2) / Math.tan((camera.fov * Math.PI / 180) / 2);
    orbit.target.copy(headPos);
    camera.position.set(headPos.x, headPos.y + viewH * 0.04, headPos.z + dist);
    orbit.update();
    return;
  }

  // Find feature meshes: noticeably smaller than the largest mesh (eyes/teeth
  // beside a head/body mesh). Union their world-space boxes into a face box.
  let maxDiag = 0;
  const meshBoxes = [];
  avatar.traverse((o) => {
    if (!o.isMesh) return;
    const b = new THREE.Box3().setFromObject(o);
    const diag = b.getSize(new THREE.Vector3()).length();
    meshBoxes.push({ b, diag });
    maxDiag = Math.max(maxDiag, diag);
  });
  const faceBox = new THREE.Box3();
  for (const { b, diag } of meshBoxes) if (diag < maxDiag * 0.5) faceBox.union(b);

  // Use the feature anchor only when it's valid and sits in the middle of the
  // model (i.e. a head scan, not a head perched atop a full body).
  const hasFeatures = !faceBox.isEmpty();
  const fc = faceBox.getCenter(new THREE.Vector3());
  const relY = size.y > 0 ? (fc.y - box.min.y) / size.y : 0.5;
  if (hasFeatures && relY < 0.7) {
    const fs = faceBox.getSize(new THREE.Vector3());
    const viewH = Math.max(fs.x, fs.y) * 1.9;
    const dist = (viewH / 2) / Math.tan((camera.fov * Math.PI / 180) / 2);
    orbit.target.set(fc.x, fc.y, fc.z);
    camera.position.set(fc.x, fc.y + viewH * 0.04, fc.z + dist);
    orbit.update();
    return;
  }

  // Fallback: size-based framing for plain heads without a Head bone.
  const fullBody = size.y > 1.0;
  const faceY = fullBody ? box.max.y - size.y * 0.06 : center.y;
  const viewH = (fullBody ? size.y * 0.18 : size.y * 1.1) * 1.15;
  const dist = (viewH / 2) / Math.tan((camera.fov * Math.PI / 180) / 2);
  orbit.target.set(center.x, faceY, center.z);
  camera.position.set(center.x, faceY, center.z + dist);
  orbit.update();
}

// ── Render loop ──────────────────────────────────────────────
const clock = new THREE.Clock();
let speakingEnv = 0;    // envelope of mouth activity → "is she talking right now?"
let silenceT = 0;       // seconds Emily has been quiet → used to time the repeat card
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  if (avatar) {
    const { open, wide, energy } = lip.update(dt);

    // "Speaking" is sustained mouth activity, not a single loud spike — so a
    // cough or a click doesn't trigger expressive brow lifts.
    speakingEnv += (open - speakingEnv) * (open > speakingEnv ? 0.3 : 0.05);
    const speaking = speakingEnv > 0.06;

    // Reveal a pending repeat-after-me only once Emily has actually gone quiet,
    // so the card appears when she STOPS talking — not when her text finished
    // generating (which is several seconds earlier, mid-speech).
    silenceT = speaking ? 0 : silenceT + dt;
    if (pendingRepeat && silenceT > 0.6) {
      console.log("[repeat] Emily silent → showing card:", pendingRepeat);
      showRepeat(pendingRepeat);
      pendingRepeat = null;
    }

    // Expression layer fills the buffer first (emotion pose, prosody liveliness,
    // blink, gaze); lip-sync mouth merges on top (max per channel).
    const { morphs, head } = face.update(dt, elapsed, { open, energy, speaking });
    mergeMax(morphs, JAW, open * 0.40);
    mergeMax(morphs, AA, open * 0.25);
    mergeMax(morphs, WIDE, Math.max(0, wide - 0.5) * 2 * open * 0.28);
    mergeMax(morphs, ROUND, Math.max(0, 0.5 - wide) * 2 * open * 0.28);
    applyMorphs(morphs);

    if (headBone) {
      headBone.rotation.x = head.rx;
      headBone.rotation.y = head.ry;
      headBone.rotation.z = head.rz;
    }
  }
  renderer.render(scene, camera);
}

// ── Resize ───────────────────────────────────────────────────
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

// ── "Repeat after me" card ───────────────────────────────────
// Emily occasionally gives the student one short corrected sentence to say
// back. Each of her lines is classified server-side (see detectRepeat); when
// it's a repeat-after-me, the exact sentence is shown here to read aloud.
let shownKey = "";        // payload currently on the card — for de-dup
let pendingRepeat = null; // { sentences, word } classified, awaiting silence

// Append a client-side event to the server's debug log (web/conversation.txt).
function logEvent(msg) {
  console.log("[event]", msg);
  fetch("/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event: msg }),
  }).catch(() => {});
}

// Wrap occurrences of `word` in the sentence with a highlight span (escaped).
function highlightWord(sentence, word) {
  const span = document.createElement("span");
  if (!word) { span.textContent = sentence; return span; }
  const re = new RegExp(`\\b(${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})\\b`, "ig");
  let last = 0, m;
  while ((m = re.exec(sentence)) !== null) {
    if (m.index > last) span.appendChild(document.createTextNode(sentence.slice(last, m.index)));
    const hl = document.createElement("span");
    hl.className = "repeat-hl";
    hl.textContent = m[0];
    span.appendChild(hl);
    last = m.index + m[0].length;
  }
  span.appendChild(document.createTextNode(sentence.slice(last)));
  return span;
}

// payload: { sentences: [string], word: string }  (word "" → plain repeat)
function showRepeat(payload) {
  const list = payload.sentences || [];
  const word = payload.word || "";
  const key = word + "|" + list.join("\n");
  // Already showing exactly this — skip so we don't re-trigger the animation.
  if (!repeatCard.hidden && key === shownKey) return;
  shownKey = key;

  // Pronunciation mode shows the target word big; plain repeat just lists lines.
  repeatLabel.textContent = word ? "🗣️ Practice this word" : "🔁 Your turn — say this";
  repeatWord.hidden = !word;
  repeatWord.textContent = word;

  repeatText.innerHTML = "";
  for (const s of list) {
    const line = document.createElement("div");
    line.className = "repeat-line";
    line.appendChild(highlightWord(s, word));
    repeatText.appendChild(line);
  }
  repeatCard.hidden = false;
  // Next frame so the display→opacity transition actually animates.
  requestAnimationFrame(() => repeatCard.classList.add("show"));
  logEvent(`card shown${word ? ` (word: ${word})` : ""}: ${list.join(" | ")}`);
}

// Ask the backend classifier whether Emily's line is a repeat-after-me, and
// show the sentence(s) if so. Deterministic — independent of the realtime model.
async function detectRepeat(text) {
  try {
    const res = await fetch("/detect-repeat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) { logEvent(`detect-repeat HTTP ${res.status}`); return; }
    const { is_repeat, sentences, focus_word } = await res.json();
    console.log("[repeat] classified:", { is_repeat, sentences, focus_word });
    // Hold the result; the render loop shows it once Emily falls silent. If
    // she's already quiet (e.g. slow classifier), it flushes on the next frame.
    if (is_repeat && sentences?.length) {
      pendingRepeat = { sentences, word: focus_word || "" };
      logEvent(`pending repeat (awaiting silence): ${sentences.join(" | ")}`);
    }
  } catch (err) {
    logEvent(`detect-repeat fetch failed: ${err.message}`);
  }
}

function hideRepeat() {
  // Hide the visible card, but DON'T discard an unshown pending sentence: a
  // freshly classified repeat should still appear even if the student just
  // made a sound (which fires a "user" transcript → hideRepeat).
  if (!repeatCard.hidden) logEvent("card hidden");
  shownKey = "";
  repeatCard.classList.remove("show");
  setTimeout(() => {
    if (!repeatCard.classList.contains("show")) repeatCard.hidden = true;
  }, 300);
}

// ── Emily state ──────────────────────────────────────────────
let emily = null;

function disconnectEmily() {
  if (!emily) return;
  logEvent("— session ended —");
  pendingRepeat = null;   // drop any unshown repeat when the session ends
  hideRepeat();
  emily.disconnect();
  emily = null;
  const btn = document.getElementById("emily-btn");
  btn.textContent = "✨ Talk to Emily";
  btn.disabled = false;
  btn.classList.remove("active");
  document.getElementById("transcript").hidden = true;
}

// ── UI wiring ────────────────────────────────────────────────
function setActiveBtn(id) {
  ["test-btn", "mic-btn"].forEach((b) =>
    document.getElementById(b).classList.toggle("active", b === id));
}

startBtn.addEventListener("click", async () => {
  startBtn.style.opacity = "0";
  setTimeout(() => (startBtn.hidden = true), 300);
  controls.hidden = false;
  await lip.unlock();
  resize();
  if (!avatar) await loadAvatar(DEFAULT_AVATAR);
});

document.getElementById("test-btn").addEventListener("click", () => {
  disconnectEmily();
  lip.useTest();
  setActiveBtn("test-btn");
  statusEl.textContent = "Test mode — fake speech";
});

document.getElementById("mic-btn").addEventListener("click", async () => {
  disconnectEmily();
  try {
    await lip.useMicrophone();
    setActiveBtn("mic-btn");
    statusEl.textContent = "Listening to your mic — talk!";
  } catch {
    statusEl.textContent = "Mic blocked (needs HTTPS or localhost)";
  }
});

document.getElementById("file-input").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  disconnectEmily();
  setActiveBtn(null);
  statusEl.textContent = `Playing ${file.name}`;
  await lip.useFile(file);
  statusEl.textContent = "Ready";
});

document.getElementById("stop-btn").addEventListener("click", () => {
  disconnectEmily();
  lip.stop();
  setActiveBtn(null);
  statusEl.textContent = "Stopped";
});

document.getElementById("load-avatar").addEventListener("click", async () => {
  const url = document.getElementById("avatar-url").value.trim();
  if (url) await loadAvatar(url);
});

document.getElementById("emily-btn").addEventListener("click", async () => {
  // Toggle: disconnect if already connected.
  if (emily) {
    disconnectEmily();
    statusEl.textContent = "Ready";
    return;
  }

  const btn = document.getElementById("emily-btn");
  btn.textContent = "⏳ Connecting…";
  btn.disabled = true;
  setActiveBtn(null);
  statusEl.textContent = "Connecting to Emily…";

  if (!avatar) await loadAvatar(DEFAULT_AVATAR);

  emily = new EmilyRealtime(lip);

  emily.onDisconnect = () => {
    disconnectEmily();
    statusEl.textContent = "Emily disconnected";
  };

  emily.onTranscript = (speaker, text) => {
    console.log(`[transcript] ${speaker}:`, text);
    // Mirror the conversation to the server's debug log (web/conversation.txt).
    fetch("/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker, text }),
    }).catch(() => {});
    // Reflect what's said on Emily's face: her own lines set her mood from
    // sentiment; while the student is talking she wears a "considering" look.
    if (speaker === "emily") {
      const emotion = classifyEmotion(text);
      face.setEmotion(emotion);
      // Nod on acknowledgment lines ("yes", "right", "exactly", positive feedback).
      if (emotion === "happy" ||
          /\b(yes|right|exactly|mhm|uh-huh|indeed|sure|of course|i see)\b/i.test(text)) {
        face.triggerNod(0.06);
      }
      // Always classify with the model: a regex would grab the first quoted
      // phrase, which is often the student's MISTAKE, not the correction.
      detectRepeat(text);
    } else if (speaker === "user") {
      // Attentive listening face + small acknowledgment nod while Emily reads the input.
      face.setEmotion("listening", 4.0);
      face.triggerNod(0.04);
      hideRepeat();   // the student has taken their turn
    }

    const area = document.getElementById("transcript");
    area.hidden = false;
    const line = document.createElement("div");
    line.className = `transcript-line ${speaker}`;
    line.textContent = `${speaker === "emily" ? "Emily" : "You"}: ${text}`;
    area.appendChild(line);
    area.scrollTop = area.scrollHeight;
  };

  try {
    await emily.connect();
    logEvent("— session started —");
    // SDP exchange complete — re-enable the button now.
    // Audio starts flowing once ICE/DTLS finish in the background (~1-2 s).
    btn.textContent = "✨ Disconnect Emily";
    btn.disabled = false;
    btn.classList.add("active");
    statusEl.textContent = "Connected — say hello in English!";
  } catch (err) {
    console.error(err);
    logEvent(`connection failed: ${err.message}`);
    statusEl.textContent = `Connection failed: ${err.message}`;
    emily = null;
    btn.textContent = "✨ Talk to Emily";
    btn.disabled = false;
  }
});

animate();

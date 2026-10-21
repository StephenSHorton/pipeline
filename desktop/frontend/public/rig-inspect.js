/**
 * Rig check for the Rig stage: shows what the rigger decided (out/<stem>/rig/inspect.json,
 * written by scripts/rig_inspect.py) on the rigged mesh already in the viewer.
 *
 * Colour modes swap the mesh to a vertex-colour material and put the textured one back on
 * close; the skeleton is drawn on top of the mesh. Bone index = skin joint order = the order
 * of inspect.json's bones, and vertex order matches rigged.glb's single primitive.
 */
const MODES = [
  ["texture", "Texture", "The painted model."],
  ["bones", "Bones", "Each part in the colour of the bone that moves it most."],
  ["heat", "One bone", "How strongly the picked bone pulls each point: red is fully, blue is not at all."],
  ["dropped", "Dropped", "Weight thrown away because the bone was too far from the point."],
  ["split", "Split", "Points whose corners follow different bones. They pull apart when the model moves."],
];
// Top looks down with the head up the screen, away from the stage bar.
const VIEWS = [["side", "Side", [1, 0, 0]], ["front", "Front", [0, 0, 1]], ["top", "Top", [0, 1, -0.001]], ["free", "3/4", [0.72, 0.38, 1]]];
const ORDER = { fail: 0, warn: 1, pass: 2 };
const MINOR = /leaf|tip|_end$/;
const NO_FIT = /leaf|tip|_end$|^ear/; // ears move few points: never a fit problem worth leading with
const LEG = /leg|foot|toe|thigh|shin|calf|knee|hoof|ball|humerus|femur|tibia/;
const ARM = /arm|hand|shoulder|clavicle|elbow|wrist|finger/;
// The rig's own steps, in the order rig.py runs them; every check belongs to one.
export const STEPS = [["plan", "Body plan"], ["rest", "Rest pose"], ["skeleton", "Skeleton"], ["skin", "Skin"], ["motion", "In motion"]];
const KEY_STEP = { planner: "plan", rest: "rest", head: "skeleton", fit: "skeleton", moved: "skeleton", facing: "skeleton", match: "skeleton",
  pose: "skeleton", pred: "skeleton", seam: "skin", folded: "skin", clips: "motion", motion: "motion", far: "motion", stretch: "motion",
  floor: "motion", unsupported: "motion" };
const PLANS = { quadruped: "Four legs", humanoid: "Person", multileg: "Many legs", avian: "Bird" };
// Why a check matters, what the rigger did about it, and what you can do (To fix). Each is a string, a map by
// level, or a function of (check, doc).
const turnedAtBirth = (doc) => !!(doc && doc.rig && doc.rig.rest && doc.rig.rest.measured && doc.rig.rest.measured.turned);
const straightened = (doc) => !!(doc && doc.rig && doc.rig.rest && doc.rig.rest.straightened);
const CHECK_HELP = {
  planner: {
    why: (c) => (c.by_hand || /set by hand$/.test(c.label || "") ? "" : "The skeleton is picked from the model's picture."),
    fix: { warn: "Pick the body plan next to Re-rig if this is wrong." },
  },
  rest: {
    why: "3D generation often turns a side-on animal's head toward the camera. The skeleton's neck runs straight, so every clip would keep the turn.",
    did: (c) => (c.straightened ? "Fitted the neck bones to the straightened head." : c.level === "warn" && !c.before_straightening ? "Rigged the head as made, because you asked." : ""),
  },
  head: {
    why: (c, doc) => (turnedAtBirth(doc) ? "3D generation turned the head toward the camera, but the neck bones run straight, so head moves swing it sideways."
      : "When the head points away from its bone, head moves swing it sideways."),
    fix: (c, doc) => (c.level === "pass" || straightened(doc) ? "" : turnedAtBirth(doc) ? "Straighten and rig, under Rest pose." : ""),
  },
  fit: {
    why: "This bone sits away from the part it moves, so that part swings around the wrong point.",
    did: (c) => (ARM.test(c.bone || "") && !LEG.test(c.bone || "") ? "Arms are aimed at the hands when rigging." : ""),
    note: (c) => (LEG.test(c.bone || "") ? "Legs aren't aimed at the feet when rigging yet, so re-rigging won't move it. It may not show in the clips." : ""),
  },
  moved: { why: "The skinning step moved some joints from where they were fitted." },
  seam: {
    why: "Many models store a point once per triangle corner, as the low-poly look does. When copies of one point follow different bones, they pull apart as it moves.",
    did: { pass: "Gave every copy of a point the same bones." },
    fix: { fail: "Re-rig. Rigs made now give every copy of a point the same bones." },
  },
  folded: { why: "A lot of weight was on root or tip bones, which don't move parts well.", did: "Moved it to their parent bones." },
  clips: { why: "These clips don't exist for this body plan." },
  facing: {
    why: "3D models don't all face the same way. A skeleton laid on backwards ties the head to tail bones and the feet to wing bones.",
    did: (c) => (c.yaw ? `Turned the skeleton ${c.yaw}° to face the way the model does.` : ""),
  },
  match: {
    why: "The skeleton for this body plan is built for a different shape, so it had to be stretched more one way than another to fit. Its clips may not suit this model.",
    did: "Fitted each leg, neck, tail and wing chain to the model's own parts.",
    fix: "If the body plan is wrong, pick another next to Re-rig.",
  },
  pose: {
    why: "Bones have to sit inside the body. Bones outside it pull the wrong parts, which the skinning step can't fix.",
    fix: { fail: "Pick another body plan next to Re-rig. This model may not fit any of our skeletons yet." },
  },
  pred: {
    why: "A skeleton predicted for this model alone shows where its legs, neck and tail really are. Our skeleton was laid onto it, so its clips still work.",
  },
  far: {
    why: "Points should move with the bones near them. Points pulled by far-away bones fling across the model when it moves, even with no tears.",
    fix: { fail: "Re-rig, or pick another body plan next to Re-rig." },
  },
  stretch: {
    why: "Edges that grow or shrink a lot smear the texture. Folded knees and elbows always squash some; above 6× it shows clearly.",
  },
  floor: {
    why: "Feet should touch the floor, and lying clips should rest on it rather than sink or float.",
    did: "Moved the body and feet to the floor in each clip.",
  },
  unsupported: {
    why: "These clips need parts this model doesn't use that way, like a flap for a bird that stands. They would throw the model into the wrong shape.",
    did: "Left them out of the rig and the FBX.",
  },
  motion: {
    why: "A tear only shows when the model moves, so every clip is played and measured.",
    fix: { fail: "Re-rig to fix it." },
  },
};
const helpText = (h, c, doc) => (typeof h === "function" ? h(c, doc) : h && typeof h === "object" ? h[c.level] : h) || "";
const MOTION_HELP = "Every clip was played on this rig and measured. Tears: points whose corners come apart. Stretch: how far the worst 1% of edges grow or shrink; low-poly knees always change some.";
// The worst 1% of edges, which the checks use; older measurements only have the single worst edge.
const stretchOf = (c) => Number(c.stretch_p99 ?? c.stretch_max);
const tidy = (n) => String(n).replace(/_/g, " ");
const cap = (x) => (x ? x[0].toUpperCase() + x.slice(1) : x);

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x) => `${Math.round(x * 100)}%`;
const deg = (x) => `${x < -0.5 ? "−" : "+"}${Math.abs(Math.round(x))}°`;
const num = (re, s) => ((String(s).match(re) || [])[1] || "");

// inspect.json's checks with a step and a short label each. Files from before v2 have neither: the step
// comes from the key, the label from the sentence. Tip bones (ears, tails' last bone) never count as a fit problem.
export function rigChecks(doc) {
  const v1 = (c) => (c.key === "seam" && c.level === "fail" ? `Tears at ${num(/(\d+)/, c.text)} points`
    : c.key === "head" && c.level !== "pass" ? `Head ${num(/(\d+)\s*°/, c.text)}° off its bone` : c.text);
  return ((doc && doc.checks) || [])
    .filter((c) => !(c.key === "fit" && NO_FIT.test(c.bone || String(c.text).split(" ")[0])))
    .map((c) => ({ ...c, step: c.step || KEY_STEP[c.key] || "skeleton", label: c.label || v1(c) }));
}

// Worst first: fail before warn, tears before anything else at that level (they are what you see), then the
// rig's step order.
const TEARS = ["seam", "motion"];
export function worstFirst(a, b) {
  const st = STEPS.map((s) => s[0]);
  const t = (c) => (TEARS.includes(c.key) ? 0 : 1);
  return ORDER[a.level] - ORDER[b.level] || t(a) - t(b) || st.indexOf(a.step || KEY_STEP[a.key]) - st.indexOf(b.step || KEY_STEP[b.key]);
}

// The finding the Rig stage leads with.
export function worstCheck(doc) {
  return rigChecks(doc).sort(worstFirst)[0] || null;
}

// viewCamera: the camera the screen shows; pixelScale: screen pixels per rendered pixel (the pixel look draws small).
// The rest are the app's hands: play a clip paused at a time, go back to the rest pose, load a glb, the original
// model's URL, rig again, rebuild the checks, pick a version, and the previous version to compare against.
export function createRigInspect({
  THREE, scene, root, camera, controls, renderer, viewCamera, pixelScale, pane, clip,
  playAt = () => {}, resume = () => {}, restPose = () => {}, loadGLB = null, originalUrl = () => null,
  rerig = () => {}, recheck = () => Promise.resolve(), goToTake = () => {}, compare = () => null,
  obstacles = () => [],
}) {
  const panel = document.createElement("aside");
  panel.className = "rig-inspect";
  panel.hidden = true;
  panel.setAttribute("aria-label", "Rig check");
  const labels = document.createElement("div");
  labels.className = "rig-inspect-labels";
  const tip = document.createElement("div");
  tip.className = "rig-inspect-tip";
  tip.hidden = true;
  pane.append(labels, tip, panel);

  // open: the check row shown in full; head: rigged | original | both; paused: { clip, t } from the In motion table.
  const S = { mode: "bones", bone: -1, skel: true, filter: "", open: "", head: "rigged", paused: null };
  let ghost = null; // the original model over (or instead of) the rig, at the rest pose
  let ghostTok = 0;
  const glbs = new Map();
  const lineMat = new THREE.LineBasicMaterial({ color: 0xf0eadc, transparent: true, opacity: 0.22, depthTest: false });
  let doc = null;
  let stem = "";
  let mesh = null;
  let saved = null; // { material, color } of the mesh before the inspector touched it
  let names = [];
  let nice = []; // names in words, for the panel, labels and hover card
  let pal = [];
  let NV = 0;
  let top = null;
  let owned = [];
  let dropped = null;
  let split = null;
  let splitPoints = 0;
  let helper = null;
  let dots = null;
  let spans = [];
  let mouse = null;
  let down = null;
  let colors = null;
  const colMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, flatShading: true });
  const grey = new THREE.Color(0.27, 0.27, 0.3);
  const cold = new THREE.Color(0.05, 0.05, 0.35);
  const heat = (t) => new THREE.Color().setHSL((1 - Math.min(1, Math.max(0, t))) * 0.6667, 1, 0.5);
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const wp = new THREE.Vector3();

  function infl(v) {
    const SI = mesh.geometry.attributes.skinIndex;
    const SW = mesh.geometry.attributes.skinWeight;
    const out = [];
    for (let k = 0; k < 4; k++) {
      const w = SW.getComponent(v, k);
      if (w > 0) out.push([SI.getComponent(v, k), w]);
    }
    return out.sort((a, b) => b[1] - a[1]);
  }

  function weightOf(v, j) {
    let w = 0;
    for (const [b, x] of infl(v)) if (b === j) w += x;
    return w;
  }

  // Copies of one point (flat-shaded corners) should follow the same bones; how far each one is off the mean.
  function cornerSplit() {
    const P = mesh.geometry.attributes.position;
    const groups = new Map();
    for (let v = 0; v < NV; v++) {
      const k = `${Math.round(P.getX(v) * 1e5)},${Math.round(P.getY(v) * 1e5)},${Math.round(P.getZ(v) * 1e5)}`;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(v);
    }
    split = new Float32Array(NV);
    splitPoints = 0;
    for (const vs of groups.values()) {
      if (vs.length < 2) continue;
      const mean = new Map();
      for (const v of vs) for (const [j, w] of infl(v)) mean.set(j, (mean.get(j) || 0) + w / vs.length);
      for (const v of vs) {
        let d = 0;
        for (const j of new Set([...mean.keys(), ...infl(v).map((x) => x[0])])) d += Math.abs(weightOf(v, j) - (mean.get(j) || 0));
        split[v] = d / 2;
      }
      if (new Set(vs.map((v) => top[v])).size > 1) splitPoints++;
    }
  }

  function bind(gltfRoot) {
    mesh = null;
    gltfRoot.traverse((o) => { if (o.isSkinnedMesh && !mesh) mesh = o; });
    if (!mesh) return false;
    const bones = mesh.skeleton.bones;
    names = doc.bones.map((b) => b.name);
    if (bones.length !== names.length || bones.some((b, i) => b.name !== names[i])) {
      console.warn("rig check: skin joints do not match inspect.json bones");
    }
    names = bones.map((b) => b.name);
    const lab = names.map((n) => cap(((doc.bones.find((b) => b.name === n) || {}).label) || tidy(n)));
    const seen = {};
    for (const l of lab) seen[l] = (seen[l] || 0) + 1;
    nice = lab.map((l, i) => (seen[l] > 1 ? `${l} ${(names[i].match(/(\d+)/) || [])[1] || ""}`.trim() : l));
    pal = names.map((_, i) => new THREE.Color().setHSL((i * 0.618034) % 1, 0.62, 0.55));
    NV = mesh.geometry.attributes.position.count;
    const tf = doc.weights && doc.weights.top_final;
    top = new Int32Array(NV);
    for (let v = 0; v < NV; v++) top[v] = tf && tf.length === NV ? tf[v] : (infl(v)[0] || [0])[0];
    owned = names.map(() => 0);
    for (let v = 0; v < NV; v++) owned[top[v]]++;
    dropped = null;
    const list = (doc.weights && doc.weights.dropped && doc.weights.dropped.list) || [];
    if (list.length) {
      dropped = new Float32Array(NV);
      for (const [v, , w] of list) if (v < NV) dropped[v] += w;
    }
    cornerSplit();
    const geo = mesh.geometry;
    saved = { material: mesh.material, color: geo.getAttribute("color") || null };
    colors = new THREE.BufferAttribute(new Float32Array(NV * 3), 3);

    helper = new THREE.SkeletonHelper(gltfRoot);
    helper.material.depthTest = false;
    helper.material.transparent = true;
    helper.renderOrder = 9;
    scene.add(helper);
    const dg = new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(names.length * 3), 3));
    dg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(names.length * 3), 3));
    dots = new THREE.Points(dg, new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, vertexColors: true, depthTest: false, transparent: true }));
    dots.renderOrder = 10;
    dots.frustumCulled = false;
    scene.add(dots);
    labels.innerHTML = "";
    spans = names.map((n) => {
      const s = document.createElement("span");
      s.textContent = nice[names.indexOf(n)];
      labels.appendChild(s);
      return s;
    });
    return true;
  }

  function paint() {
    if (!mesh) return;
    const geo = mesh.geometry;
    if (S.mode === "texture") {
      mesh.material = saved.material;
      if (saved.color) geo.setAttribute("color", saved.color);
      else geo.deleteAttribute("color");
      return;
    }
    const c = new THREE.Color();
    for (let v = 0; v < NV; v++) {
      if (S.mode === "bones") {
        c.copy(pal[top[v]]);
        if (S.bone >= 0 && top[v] !== S.bone) c.lerp(grey, 0.85);
      } else if (S.mode === "heat") {
        const w = S.bone >= 0 ? weightOf(v, S.bone) : 0;
        c.copy(S.bone < 0 ? grey : w > 0 ? heat(w) : cold);
      } else if (S.mode === "dropped") {
        c.copy(dropped && dropped[v] ? new THREE.Color(1, 1 - Math.min(1, dropped[v] * 3), 0) : grey);
      } else {
        c.copy(split[v] > 0.05 ? heat(Math.min(1, split[v] * 2)) : grey);
      }
      colors.setXYZ(v, c.r, c.g, c.b);
    }
    colors.needsUpdate = true;
    geo.setAttribute("color", colors);
    mesh.material = colMat;
  }

  function colorSkeleton() {
    if (!helper) return;
    const ca = helper.geometry.attributes.color;
    const hot = new THREE.Color(1, 0.35, 0.3);
    const dim = new THREE.Color(0.45, 0.45, 0.48);
    const ink = new THREE.Color(0.94, 0.92, 0.86);
    let i = 0;
    for (const b of helper.bones) {
      if (!b.parent || !b.parent.isBone) continue;
      const k = names.indexOf(b.name);
      const pk = names.indexOf(b.parent.name);
      const col = k === S.bone || pk === S.bone ? hot : b.parent.name === "root" ? dim : ink;
      ca.setXYZ(i++, col.r, col.g, col.b);
      ca.setXYZ(i++, col.r, col.g, col.b);
    }
    ca.needsUpdate = true;
  }

  function screenOf(i, cam) {
    mesh.skeleton.bones[i].getWorldPosition(wp);
    wp.project(cam);
    return [(wp.x + 1) / 2 * pane.clientWidth, (1 - wp.y) / 2 * pane.clientHeight, wp.z < 1];
  }

  function drawSkeleton() {
    const on = S.skel;
    helper.visible = dots.visible = on;
    labels.hidden = !on;
    if (!on) return;
    const pa = dots.geometry.attributes.position;
    const ca = dots.geometry.attributes.color;
    mesh.skeleton.bones.forEach((b, i) => {
      b.getWorldPosition(wp);
      pa.setXYZ(i, wp.x, wp.y, wp.z);
      const col = i === S.bone ? new THREE.Color(1, 0.25, 0.2) : pal[i];
      ca.setXYZ(i, col.r, col.g, col.b);
    });
    pa.needsUpdate = ca.needsUpdate = true;
    dots.material.size = Math.max(2, 7 / Math.max(1, pixelScale()));
    // Names: the picked bone first, then skip any that would land on one already placed.
    const cam = viewCamera();
    const placed = [];
    const order = [...names.keys()].sort((a, b) => (b === S.bone) - (a === S.bone));
    const near = (i) => S.bone < 0 || i === S.bone || doc.bones[i]?.parent === S.bone || doc.bones[S.bone]?.parent === i;
    for (const i of order) {
      const s = spans[i];
      let show = (!MINOR.test(names[i]) && near(i)) || i === S.bone;
      const [x, y, front] = show ? screenOf(i, cam) : [0, 0, false];
      const w = nice[i].length * 5.6 + 6;
      show = show && front && (i === S.bone || !placed.some(([px, py, pw]) => x < px + pw && px < x + w && Math.abs(py - y) < 11));
      s.hidden = !show;
      if (!show) continue;
      placed.push([x, y, w]);
      s.style.left = `${x}px`;
      s.style.top = `${y}px`;
      s.classList.toggle("is-picked", i === S.bone);
    }
  }

  function hitVertex(clientX, clientY) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((clientX - r.left) / Math.max(1, r.width)) * 2 - 1, -((clientY - r.top) / Math.max(1, r.height)) * 2 + 1);
    ray.setFromCamera(ndc, viewCamera());
    const h = ray.intersectObject(mesh, false)[0];
    if (!h || !h.face) return -1;
    const p = new THREE.Vector3();
    let best = h.face.a;
    let bd = Infinity;
    for (const v of [h.face.a, h.face.b, h.face.c]) {
      mesh.getVertexPosition(v, p);
      mesh.localToWorld(p);
      const d = p.distanceTo(h.point);
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  }

  function showTip() {
    if (!mouse || !mesh || !mesh.visible) { tip.hidden = true; return; }
    const v = hitVertex(mouse[0], mouse[1]);
    if (v < 0) { tip.hidden = true; return; }
    const rows = infl(v).map(([j, w]) => `<div class="rig-inspect-row"><span class="rig-inspect-sw" style="background:#${pal[j].getHexString()}"></span>${esc(nice[j])}<b>${w.toFixed(2)}</b></div>`);
    let extra = "";
    if (dropped && dropped[v]) extra += `<div class="is-warn">${dropped[v].toFixed(2)} weight dropped</div>`;
    if (split[v] > 0.1) extra += `<div class="is-fail">Corners disagree by ${pct(split[v])}</div>`;
    tip.innerHTML = `<div class="rig-inspect-tip-title">Point ${v}</div>${rows.join("")}${extra}<div class="rig-inspect-mute">Click to pick ${esc(nice[infl(v)[0][0]])}</div>`;
    const r = pane.getBoundingClientRect();
    tip.style.left = `${Math.min(mouse[0] - r.left + 14, pane.clientWidth - 200)}px`;
    tip.style.top = `${mouse[1] - r.top + 14}px`;
    tip.hidden = false;
  }

  // From the list a second click lets go of the bone; from the model it always picks.
  function pick(i, toggle) {
    S.bone = toggle && S.bone === i ? -1 : i;
    if (S.bone >= 0 && S.mode !== "bones" && S.mode !== "heat") S.mode = "heat";
    paint();
    colorSkeleton();
    renderPanel();
    panel.querySelector(".rig-inspect-bones .is-picked")?.scrollIntoView({ block: "nearest" });
  }

  // The part of the pane the model can be seen in, in NDC: left of this panel and above the stage bar.
  function freeArea() {
    const pr = pane.getBoundingClientRect();
    let { left, right, top, bottom } = pr;
    if (!panel.hidden) {
      const r = panel.getBoundingClientRect();
      if (r.width && r.left > pr.left + pr.width / 2) right = Math.min(right, r.left);
    }
    for (const el of obstacles()) {
      const r = el && el.getBoundingClientRect();
      if (!r || !r.width || !r.height || r.left > right) continue;
      if (r.top > pr.top + pr.height / 2) bottom = Math.min(bottom, r.top - 8);
      else top = Math.max(top, r.bottom + 8);
    }
    const nx = (x) => ((x - pr.left) / Math.max(1, pr.width)) * 2 - 1;
    const ny = (y) => 1 - ((y - pr.top) / Math.max(1, pr.height)) * 2;
    return { x0: nx(left), x1: nx(Math.max(left + 80, right)), y0: ny(Math.max(top + 80, bottom)), y1: ny(top) };
  }

  // Points of the mesh as it is posed now, in world space (all of them, or those a bone moves most).
  function posedBox(bone) {
    root.updateMatrixWorld(true);
    const box = new THREE.Box3();
    const p = new THREE.Vector3();
    const step = bone >= 0 ? 1 : Math.max(1, Math.floor(NV / 4000));
    for (let v = 0; v < NV; v += step) {
      if (bone >= 0 && top[v] !== bone) continue;
      mesh.getVertexPosition(v, p);
      box.expandByPoint(mesh.localToWorld(p));
    }
    return box;
  }

  // A bone's part with room around it to see where it sits on the model.
  function boneBox(i) {
    const box = posedBox(i);
    for (const k of [i, ...doc.bones.map((b, c) => (b.parent === i ? c : -1)).filter((c) => c >= 0)]) {
      mesh.skeleton.bones[k].getWorldPosition(wp);
      box.expandByPoint(wp);
    }
    // At least 3/4 of the model across, kept on the model rather than centred on empty floor.
    const all = posedBox(-1);
    const as = all.getSize(new THREE.Vector3());
    const min = 0.75 * Math.max(as.x, as.y, as.z);
    const out = box.clone();
    for (const ax of ["x", "y", "z"]) {
      const want = Math.min(Math.max(box.max[ax] - box.min[ax], min), as[ax]);
      if (box.max[ax] - box.min[ax] >= want) continue;
      const c = Math.min(Math.max((box.min[ax] + box.max[ax]) / 2, all.min[ax] + want / 2), all.max[ax] - want / 2);
      out.min[ax] = c - want / 2;
      out.max[ax] = c + want / 2;
    }
    return out;
  }

  // Look from a view (or the current direction) and fit box into the free part of the pane.
  function frameBox(box, id) {
    if (box.isEmpty()) return;
    const dir = Array.isArray(id) ? new THREE.Vector3(...id).normalize() : id ? new THREE.Vector3(...VIEWS.find((v) => v[0] === id)[2]).normalize()
      : camera.position.clone().sub(controls.target).normalize();
    const c = box.getCenter(new THREE.Vector3());
    camera.position.copy(c).add(dir);
    camera.lookAt(c);
    camera.updateMatrixWorld(true);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    let hx = 0, hy = 0, hz = 0;
    for (let k = 0; k < 8; k++) {
      const q = new THREE.Vector3(k & 1 ? box.max.x : box.min.x, k & 2 ? box.max.y : box.min.y, k & 4 ? box.max.z : box.min.z).sub(c);
      hx = Math.max(hx, Math.abs(q.dot(right)));
      hy = Math.max(hy, Math.abs(q.dot(up)));
      hz = Math.max(hz, q.dot(dir));
    }
    const f = freeArea();
    const tanV = Math.tan(THREE.MathUtils.degToRad((camera.fov || 40) / 2));
    const tanH = tanV * (camera.aspect || 1);
    const m = 0.86;
    const d = hz + Math.max(hx / (((f.x1 - f.x0) / 2) * tanH * m), hy / (((f.y1 - f.y0) / 2) * tanV * m), 0.05);
    const cx = (f.x0 + f.x1) / 2;
    const cy = (f.y0 + f.y1) / 2;
    controls.target.copy(c).addScaledVector(right, -cx * d * tanH).addScaledVector(up, -cy * d * tanV);
    controls.autoRotate = false;
    controls.minDistance = Math.min(controls.minDistance, d * 0.5);
    controls.maxDistance = Math.max(controls.maxDistance, d * 2);
    camera.position.copy(controls.target).addScaledVector(dir, d);
    camera.lookAt(controls.target);
    controls.update();
  }

  // A view of the whole model, fitted to the free part of the pane.
  function setView(id) {
    if (mesh) frameBox(posedBox(-1), id);
  }

  function legend() {
    const sel = S.bone >= 0 ? `<b title="${esc(names[S.bone])}">${esc(nice[S.bone])}</b>` : "";
    const w = doc.weights || {};
    if (S.mode === "heat") return S.bone < 0 ? `<span class="is-warn">Pick a bone in the list or click the model.</span>` : `${sel}: red pulls fully, blue a little, dark blue not at all.<div class="rig-inspect-ramp"></div>`;
    if (S.mode === "bones") return `Each part takes the colour of the bone that moves it most.${sel ? ` ${sel} is lit, the rest dimmed.` : ""}`;
    if (S.mode === "dropped") {
      const total = w.dropped && w.dropped.total;
      if (!dropped) return total ? `${total} weights were dropped, but this version did not record where. Re-rig to see them.` : "Nothing was dropped.";
      return `Yellow to red: weight thrown away because the bone was too far. ${total} in all.`;
    }
    if (S.mode === "split") {
      const raw = w.seam_raw && w.seam_raw.positions_with_conflicting_top_bone;
      const before = raw != null && raw !== splitPoints ? ` Straight from the skinner it was ${raw}.` : "";
      return splitPoints ? `<span class="is-fail">${splitPoints} points have corners on different bones.</span> They tear when the model moves. Re-rig to fix.${before}` : `No split points. Corners at the same point move together.${before}`;
    }
    return "The painted model.";
  }

  function boneCard() {
    if (S.bone < 0) return `<p class="rig-inspect-mute">Pick a bone to see what it does.</p>`;
    const nm = names[S.bone];
    const b = doc.bones.find((x) => x.name === nm) || {};
    const k = doc.bones.indexOf(b);
    const f = (doc.fit || []).find((x) => x.bone === nm);
    const c = doc.clips && doc.clips[clip()];
    const peak = c && k >= 0 ? c.peak_deg[k] : null;
    const parent = b.parent >= 0 && doc.bones[b.parent] ? nice[names.indexOf(doc.bones[b.parent].name)] || doc.bones[b.parent].name : "";
    const off = f ? f.centroid_off_bone_rel_h : null;
    const row = (k2, v, cls) => `<tr><td>${k2}</td><td${cls ? ` class="${cls}"` : ""}>${v}</td></tr>`;
    return `<div class="rig-inspect-card"><div class="rig-inspect-card-title" title="${esc(nm)}"><span class="rig-inspect-sw" style="background:#${pal[S.bone].getHexString()}"></span>${esc(nice[S.bone])}</div><table>
      ${parent ? row("Hangs from", esc(parent)) : ""}
      ${row("Driven by", b.template ? `template bone ${esc(b.template)}` : `<span class="is-warn">${esc(b.rule || "nothing")}</span>`)}
      ${b.ref_vs_rest_deg != null ? row("Turned in the start pose", `${b.ref_vs_rest_deg}°`) : ""}
      ${row("Points it moves most", owned[S.bone])}
      ${f ? row("Usual distance to its points", `${pct(f.median_dist_rel_h)} of height`) : ""}
      ${off != null ? row("Its part's centre is off by", `${pct(off)} of height`, off > 0.12 ? "is-warn" : "") : ""}
      ${peak != null ? row(`Biggest turn in ${esc(tidy(clip()))}`, `${peak}°`) : ""}
    </table></div>`;
  }


  // Head: the original model against the rig, both at the rest pose and seen from the top.
  const rest = () => (doc && doc.rig && doc.rig.rest) || null;
  const headUI = () => !!(loadGLB && rest() && rest().measured && originalUrl());

  function clearGhost() {
    ghostTok++;
    if (ghost) {
      if (ghost.parent) ghost.parent.remove(ghost);
      ghost.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
      ghost = null;
    }
    if (mesh) mesh.visible = true;
  }

  async function setHead(m) {
    if (!mesh || m === S.head) return;
    clearGhost();
    S.head = m;
    if (m === "rigged") {
      resume();
      renderPanel();
      return;
    }
    restPose();
    mesh.skeleton.pose();
    S.paused = null;
    setView("top", true);
    renderPanel();
    const tok = ghostTok;
    const url = originalUrl();
    try {
      if (!glbs.has(url)) glbs.set(url, loadGLB(url));
      const g = await glbs.get(url);
      if (tok !== ghostTok || !mesh || !mesh.parent) return;
      ghost = new THREE.Group();
      ghost.position.copy(mesh.position);
      ghost.quaternion.copy(mesh.quaternion);
      ghost.scale.copy(mesh.scale);
      g.scene.updateMatrixWorld(true);
      g.scene.traverse((o) => {
        if (!o.isMesh) return;
        const geo = o.geometry.clone().applyMatrix4(o.matrixWorld);
        if (m === "original") { ghost.add(new THREE.Mesh(geo, o.material)); return; }
        const w = new THREE.LineSegments(new THREE.WireframeGeometry(geo), lineMat);
        w.renderOrder = 8;
        ghost.add(w);
        geo.dispose();
      });
      mesh.parent.add(ghost);
      mesh.visible = m !== "original";
    } catch (err) {
      console.warn("rig check: could not load the original model", err);
      S.head = "rigged";
      resume();
      renderPanel();
    }
  }

  function headBlock() {
    const r = rest();
    if (!r || !r.measured) return "";
    const m = r.measured;
    const st = r.straightened;
    const read = st ? `Head turned ${deg(m.head_yaw_deg)} → ${deg(st.head_yaw_deg)}`
      : m.turned && r.before_straightening ? `Head turned ${deg(m.head_yaw_deg)} when it was made`
        : m.turned ? `Head turned ${deg(m.head_yaw_deg)}, kept as made` : `Head straight (${Math.abs(Math.round(m.head_yaw_deg))}°)`;
    const seg = st && headUI()
      ? `<div class="dock-group rig-head-seg">${[["rigged", "Rigged", "The rig as it moves."], ["original", "Original", "Your model as it was made."], ["both", "Both", "Your model as a wireframe over the rig."]]
        .map(([id, l, t]) => `<button type="button" data-head="${id}" aria-pressed="${S.head === id}" title="${t}">${l}</button>`).join("")}</div>` : "";
    const btn = st ? `<button type="button" class="takes-act" data-rerig="off" title="Rigs again with the head as the model has it. This version stays in Versions.">Rig it as made</button>`
      : m.turned ? `<button type="button" class="takes-act" data-rerig="auto" title="Rigs again with the head straightened. This version stays in Versions.">Straighten and rig</button>` : "";
    const kept = st ? `<p class="rig-inspect-mute">Your model file is never changed.</p>` : "";
    return `<div class="rig-head">${seg}<div class="rig-head-read">${read}</div>${kept}${btn}</div>`;
  }

  // Show me: turn on the view that proves a check. A string back is a part of the panel to scroll to.
  // The side a bone is on: a right-hand bone is seen from the right.
  const sideOf = (n) => (/(^|_)r$|right/i.test(n || "") ? [-1, 0, 0] : "side");

  function focusBone(n, view, mode = "heat") {
    const i = names.indexOf(n);
    if (i < 0) return;
    S.bone = i;
    S.mode = mode;
    S.skel = true;
    frameBox(boneBox(i), view);
  }

  function atMoment(name, t, tear, bone) {
    if (S.head !== "rigged") setHead("rigged");
    playAt(name, t || 0);
    S.paused = { clip: name, t: t || 0 };
    if (tear) {
      S.mode = "split";
      S.skel = false;
      frameBox(posedBox(-1), "free");
    } else if (bone && names.includes(bone)) {
      // The stretched part, from the side, where legs and their bends show.
      S.bone = names.indexOf(bone);
      S.mode = "heat";
      frameBox(boneBox(S.bone), sideOf(bone));
    }
  }

  function moment(name) {
    const c = doc.motion && doc.motion.clips && doc.motion.clips[name];
    if (!c) return;
    if (c.tears > 0) atMoment(name, c.tear_t ?? c.worst_t, true);
    else atMoment(name, c.worst_t, false, c.worst_bone);
    paint();
    colorSkeleton();
    renderPanel();
  }

  // A head that wasn't straightened: the rig as painted, from the top, with the head bone lit against the spine.
  function showTurn() {
    if (S.head !== "rigged") setHead("rigged");
    restPose();
    mesh.skeleton.pose();
    S.paused = null;
    focusBone("head", "top", "texture");
    setView("top");
  }

  const SHOW = {
    planner: () => ".rig-inspect-plan",
    rest: (c) => {
      if (rest() && rest().straightened) { S.mode = "texture"; setHead("both"); } else showTurn();
    },
    head: () => focusBone("head", "top"),
    fit: (c) => focusBone(c.bone, sideOf(c.bone)),
    moved: () => { S.skel = true; },
    seam: () => { S.mode = "split"; S.skel = false; setView("front"); },
    folded: () => { S.mode = "dropped"; },
    motion: (c) => (c.level === "fail" && c.clip ? atMoment(c.clip, c.t, true) : ".rig-inspect-motion"),
    stretch: (c) => {
      const m = doc.motion && doc.motion.clips && doc.motion.clips[c.clip];
      return m ? atMoment(c.clip, m.worst_t, false, m.worst_bone) : ".rig-inspect-motion";
    },
    far: () => ".rig-inspect-motion",
    floor: () => ".rig-inspect-motion",
    pose: () => { S.skel = true; },
    facing: () => { S.skel = true; setView("side"); },
    pred: () => { S.skel = true; },
  };
  const canShow = (c) => !!SHOW[c.key]
    && (c.key !== "rest" || (rest() && rest().straightened ? headUI() : !!(rest() && rest().measured && rest().measured.turned && names.includes("head"))))
    && (!["head", "fit"].includes(c.key) || names.includes(c.bone || "head"))
    && (c.key !== "motion" || (c.level === "fail" ? !!c.clip : !!doc.motion))
    && (c.key !== "folded" || !!dropped);

  function show(c) {
    if (c.key !== "rest" && S.head !== "rigged") setHead("rigged");
    const to = SHOW[c.key](c);
    paint();
    colorSkeleton();
    renderPanel();
    if (typeof to === "string") panel.querySelector(to)?.scrollIntoView({ block: "nearest" });
  }

  function checkRow(c) {
    const open = S.open === c.key;
    const h = CHECK_HELP[c.key] || {};
    const [why, did, fix, note] = ["why", "did", "fix", "note"].map((k) => helpText(h[k], c, doc));
    const same = (a, b) => a.replace(/\W+/g, " ").trim().toLowerCase() === b.replace(/\W+/g, " ").trim().toLowerCase();
    return `<div class="rig-check is-${c.level}${open ? " is-open" : ""}">
      <button type="button" class="rig-inspect-chip is-${c.level}" data-check="${esc(c.key)}" aria-expanded="${open}" title="${esc(c.text)}">${esc(c.label)}</button>
      ${open ? `<div class="rig-check-body">
        ${c.text !== c.label ? `<p>${esc(c.text)}</p>` : ""}
        ${why && !same(why, c.text || "") ? `<p><b>Why:</b> ${esc(why)}</p>` : ""}
        ${did ? `<p><b>What we did:</b> ${esc(did)}</p>` : ""}
        ${fix ? `<p><b>To fix:</b> ${esc(fix)}</p>` : ""}
        ${note ? `<p class="rig-inspect-mute">${esc(note)}</p>` : ""}
        ${c.key === "rest" ? headBlock() : ""}
        ${canShow(c) ? `<button type="button" class="takes-act" data-show="${esc(c.key)}">Show me</button>` : ""}
      </div>` : ""}
    </div>`;
  }

  function motionBlock() {
    const mo = doc.motion;
    const head = `<div class="rail-label" title="${esc(MOTION_HELP)}">In motion</div>`;
    if (!mo || !mo.clips || !Object.keys(mo.clips).length) {
      return `${head}<div class="rig-inspect-motion"><p class="rig-inspect-mute">Not measured for this version.</p>
        ${S.checking ? `<p class="rig-inspect-mute">Checking…</p>` : `<button type="button" class="takes-act" data-measure title="Play every clip on this rig and measure it. Takes a few seconds.">Measure</button>`}</div>`;
    }
    const clips = Object.entries(mo.clips);
    const rows = clips.map(([n, c]) => `<tr data-moment="${esc(n)}" class="${S.paused && S.paused.clip === n ? "is-picked" : ""}">
      <td><button type="button" data-moment="${esc(n)}" title="${c.tears ? `Pause ${esc(tidy(n))} where it tears most` : `Pause ${esc(tidy(n))} at its worst stretch`}">${esc(tidy(n))}</button></td>
      <td class="${c.tears ? "is-fail" : ""}">${c.tears}</td><td>${stretchOf(c).toFixed(1)}×</td></tr>`).join("");
    const paused = S.paused ? `<div class="rig-inspect-paused">Paused at ${S.paused.t.toFixed(1)} s in ${esc(tidy(S.paused.clip))} · <button type="button" class="rig-inspect-link" data-resume>Play</button></div>` : "";
    const lines = [];
    // Tears at rest and tears in a clip are counted differently; say how they relate.
    const seam = rigChecks(doc).find((c) => c.key === "seam" && c.level === "fail");
    const most = clips.reduce((a, b) => (b[1].tears > a[1].tears ? b : a));
    if (seam && most[1].tears) lines.push(`${seam.points ?? splitPoints} points can tear. How many come apart depends on the move: up to ${most[1].tears} in ${tidy(most[0])}.`);
    // Stretch that sits on a bone the Skeleton step flagged.
    const fit = rigChecks(doc).find((c) => c.key === "fit" && c.level !== "pass" && c.bone);
    const on = fit ? clips.filter(([, c]) => !c.tears && c.worst_bone === fit.bone).map(([n]) => tidy(n)) : [];
    if (on.length) {
      const list = on.length > 1 ? `${on.slice(0, -1).join(", ")} and ${on[on.length - 1]}` : on[0];
      lines.push(`The worst stretch in ${list} is at the ${(doc.bones.find((b) => b.name === fit.bone) || {}).label || tidy(fit.bone)}, the bone flagged under Skeleton.`);
    }
    lines.push(clips.some(([, c]) => c.stretch_p99 != null)
      ? "Stretch: how much the worst 1% of edges grow or shrink. Lower is better; above 3× it starts to show, above 6× it fails."
      : "Stretch: how much the edge that changes most grows or shrinks. Lower is better; knees and elbows always stretch some.");
    return `${head}<div class="rig-inspect-motion"><table class="rig-motion"><thead><tr title="${esc(MOTION_HELP)}"><th>Clip</th><th>Tears</th><th>Stretch</th></tr></thead><tbody>${rows}</tbody></table>${paused}
      ${lines.map((l) => `<p class="rig-inspect-mute rig-motion-note">${esc(l)}</p>`).join("")}</div>`;
  }

  // Lower is better: the better side green, the worse amber.
  function pair(a, b, as = a, bs = b, arrow = true) {
    const cls = (x, y) => (x < y ? "is-better" : x > y ? "is-worse" : "");
    const mark = !arrow ? "" : b < a ? " ▼" : b > a ? " ▲" : "";
    return `<span class="${cls(a, b)}">${as}</span> → <span class="${cls(b, a)}">${bs}${mark}</span>`;
  }

  function headWords(h) {
    if (!h) return ["not measured", 0];
    const n = (x) => Math.abs(Math.round(x || 0));
    if (h.straightened) return [`straightened ${n(h.before)}°`, n(h.off_deg ?? h.after)];
    if (h.made_before && h.off_deg != null && n(h.off_deg) > 8) return [`${n(h.off_deg)}° off`, n(h.off_deg)];
    if (h.before != null && n(h.before) > 8) return [`kept turned ${n(h.before)}°`, n(h.off_deg ?? h.before)];
    if (h.off_deg != null && n(h.off_deg) > 8) return [`${n(h.off_deg)}° off`, n(h.off_deg)];
    return ["straight", n(h.off_deg)];
  }

  function compareBlock() {
    const c = compare();
    if (!c) return "";
    const head = `<div class="rail-label">Since version ${c.n}</div>`;
    const back = `<button type="button" class="takes-act" data-take="${esc(c.id)}">Go back to version ${c.n}</button>`;
    if (!c.summary) return `${head}<p class="rig-inspect-mute">Version ${c.n} was made before rig checks.</p>${back}`;
    const a = c.summary;
    const b = c.current || doc.summary || {};
    const rows = [];
    if (typeof a.tears === "number" && typeof b.tears === "number") rows.push(["Tears", pair(a.tears, b.tears)]);
    if (a.head || b.head) {
      const [aw, ar] = headWords(a.head);
      const [bw, br] = headWords(b.head);
      rows.push(["Head", aw === bw ? "same" : pair(ar, br, aw, bw, false)]);
    }
    // Stretch per clip: the two that changed most, the rest behind a toggle.
    const both = Object.entries(b.motion || {}).filter(([n]) => a.motion && a.motion[n])
      .map(([n, m]) => [n, a.motion[n].stretch_max, m.stretch_max])
      .sort((x, y) => Math.abs(Math.log(y[2] / y[1])) - Math.abs(Math.log(x[2] / x[1])));
    const shown = S.allClips ? both : both.slice(0, 2);
    for (const [n, o, m] of shown) rows.push([`${esc(tidy(n))} stretch`, pair(o, m, `${o.toFixed(1)}×`, `${m.toFixed(1)}×`)]);
    const more = both.length > 2 ? `<button type="button" class="rig-inspect-link rig-compare-more" data-all-clips>${S.allClips ? "Fewer clips" : `All ${both.length} clips`}</button>` : "";
    if (a.class && b.class) rows.push(["Body plan", a.class === b.class ? "same" : `${esc(PLANS[a.class] || a.class)} → ${esc(PLANS[b.class] || b.class)}`]);
    rows.push(["Model", c.sameModel ? "same model" : "made from a different model"]);
    return `${head}<table class="rig-compare">${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>${more}${back}`;
  }

  const found = (i) => !S.filter || names[i].includes(S.filter) || nice[i].toLowerCase().includes(S.filter.toLowerCase());

  function renderPanel() {
    if (!doc) return;
    const checks = rigChecks(doc);
    const depth = (i) => { let d = 0; let p = doc.bones[i] && doc.bones[i].parent; while (p >= 0 && d < 12) { d++; p = doc.bones[p].parent; } return d; };
    const pl = doc.planner || {};
    const t = doc.template || {};
    const fmt = (x, n = 3) => (typeof x === "number" ? x.toFixed(n) : "?");
    const scroll = panel.querySelector(".rig-inspect-body")?.scrollTop || 0;
    const steps = STEPS.map(([id, title]) => {
      const cs = checks.filter((c) => c.step === id);
      return cs.length ? `<div class="rig-step"><div class="rig-step-title">${title}</div>${cs.map(checkRow).join("")}</div>` : "";
    }).join("");
    panel.innerHTML = `
      <div class="rig-inspect-bar"><span>Rig check</span><button type="button" class="rig-inspect-close" title="Close the rig check" aria-label="Close">×</button></div>
      <div class="rig-inspect-body">
        <div class="rail-label">How it was rigged</div>
        <div class="rig-inspect-checks">${steps}</div>
        ${motionBlock()}
        ${compareBlock()}
        <div class="rail-label">Colour</div>
        <div class="dock-group">${MODES.map(([id, label, title]) => `<button type="button" data-mode-rig="${id}" aria-pressed="${S.mode === id}" title="${esc(title)}">${label}</button>`).join("")}</div>
        <div class="blurb rig-inspect-legend">${legend()}</div>
        <div class="rail-label">Show</div>
        <div class="dock-group">
          <button type="button" data-skel aria-pressed="${S.skel}" title="Draw the bones and their names on top of the model.">Skeleton</button>
          ${VIEWS.map(([id, label]) => `<button type="button" data-view="${id}" title="Look from the ${label.toLowerCase()}.">${label}</button>`).join("")}
          ${headUI() && rest().straightened ? `<button type="button" data-ghost aria-pressed="${S.head === "both"}" title="Your model as it was made, as a wireframe over the rig at rest.">Original</button>` : ""}
        </div>
        <div class="rail-label">Bones · ${names.length}</div>
        <input type="search" class="rig-inspect-filter" placeholder="Find a bone" value="${esc(S.filter)}" spellcheck="false" />
        <div class="rig-inspect-bones">${names.map((n, i) => `<button type="button" data-bone="${i}" class="${i === S.bone ? "is-picked" : ""}" title="${esc(n)}" style="padding-left:${6 + Math.min(depth(i), 8) * 8}px"${S.filter && !found(i) ? " hidden" : ""}><span class="rig-inspect-sw" style="background:#${pal[i].getHexString()}"></span>${esc(nice[i])}<i>${owned[i] || ""}</i></button>`).join("")}</div>
        ${boneCard()}
        <div class="rail-label">Body plan</div>
        <div class="rig-inspect-plan">
          ${pl.image ? `<img alt="" src="/out/${encodeURIComponent(stem)}/${esc(pl.image)}" />` : ""}
          <table>
            <tr><td>Chosen</td><td>${esc(PLANS[pl.choice] || pl.choice || PLANS[doc.rig && doc.rig.class] || "?")}</td></tr>
            <tr><td>How sure</td><td${pl.confidence != null && pl.confidence < 0.6 ? ' class="is-warn"' : ""}>${pl.confidence != null ? pct(pl.confidence) : pl.unrecorded ? "Not recorded" : "Set by hand"}</td></tr>
          </table>
        </div>
        <details class="rig-inspect-more"${S.details ? " open" : ""}><summary class="rail-label">Skeleton fit</summary>
        <table>
          <tr><td>Template</td><td>${esc(String(t.rig || "?").split("/").pop().replace(/\.glb$/, ""))}</td></tr>
          <tr><td>Start pose</td><td>${esc(t.ref_clip || "?")}, first frame</td></tr>
          <tr><td>Scale</td><td>${fmt(t.fit_scale)}</td></tr>
          <tr><td>Offset (m)</td><td>${(t.fit_offset || []).map((x) => fmt(x)).join(", ")}</td></tr>
          <tr><td>Joints moved by skinning</td><td>${fmt(t.max_joint_move_by_skinner_m, 2)} m</td></tr>
          ${doc.head ? `<tr><td>Head vs head bone</td><td class="${Math.abs(doc.head.off_deg) > 15 ? "is-fail" : Math.abs(doc.head.off_deg) > 6 ? "is-warn" : ""}">${Math.abs(doc.head.off_deg).toFixed(0)}° apart</td></tr>` : ""}
        </table></details>
        ${(doc.notes || []).length ? `<div class="rail-label">What the rigger did</div><ul class="rig-inspect-notes">${doc.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
      </div>`;
    panel.querySelector(".rig-inspect-body").scrollTop = scroll;
  }

  async function measure() {
    S.checking = true;
    renderPanel();
    try { await recheck(); } finally { S.checking = false; renderPanel(); }
  }

  panel.addEventListener("click", (e) => {
    const el = e.target.closest("button, tr[data-moment]");
    if (!el) return;
    const d = el.dataset;
    if (el.classList.contains("rig-inspect-close")) { api.onClose && api.onClose(); return; }
    if (d.modeRig) {
      S.mode = d.modeRig;
      paint();
      renderPanel();
    } else if (el.hasAttribute("data-skel")) {
      S.skel = !S.skel;
      renderPanel();
    } else if (d.view) {
      setView(d.view);
    } else if (d.bone) {
      pick(Number(d.bone), true);
    } else if (d.check) {
      S.open = S.open === d.check ? "" : d.check;
      renderPanel();
    } else if (d.show) {
      const c = rigChecks(doc).find((x) => x.key === d.show);
      if (c) show(c);
    } else if (d.head) {
      setHead(d.head);
    } else if (el.hasAttribute("data-ghost")) {
      setHead(S.head === "both" ? "rigged" : "both");
    } else if (d.rerig) {
      rerig({ straighten: d.rerig });
    } else if (el.hasAttribute("data-measure")) {
      measure();
    } else if (d.moment) {
      moment(d.moment);
    } else if (el.hasAttribute("data-resume")) {
      S.paused = null;
      resume();
      renderPanel();
    } else if (el.hasAttribute("data-all-clips")) {
      S.allClips = !S.allClips;
      renderPanel();
    } else if (d.take) {
      goToTake(d.take);
    }
  });
  panel.addEventListener("toggle", (e) => { if (e.target.classList.contains("rig-inspect-more")) S.details = e.target.open; }, true);
  panel.addEventListener("input", (e) => {
    if (!e.target.classList.contains("rig-inspect-filter")) return;
    S.filter = e.target.value.trim();
    panel.querySelectorAll(".rig-inspect-bones [data-bone]").forEach((b) => { b.hidden = !!S.filter && !found(Number(b.dataset.bone)); });
  });
  const canvas = renderer.domElement;
  canvas.addEventListener("pointermove", (e) => { if (mesh && !down) mouse = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerleave", () => { mouse = null; tip.hidden = true; });
  canvas.addEventListener("pointerdown", (e) => { if (mesh) { down = [e.clientX, e.clientY]; tip.hidden = true; } });
  canvas.addEventListener("pointerup", (e) => {
    const was = down;
    down = null;
    if (!mesh || !mesh.visible || !was || Math.hypot(e.clientX - was[0], e.clientY - was[1]) > 4) return;
    const v = hitVertex(e.clientX, e.clientY);
    if (v >= 0) pick(infl(v)[0][0]);
  });

  const api = {
    onClose: null,
    isOpen: () => !!mesh,
    // Show the check for the rigged scene in the viewer. inspectDoc is inspect.json; focus is a check key
    // to open with its view turned on.
    open(inspectDoc, assetStem, opts) {
      api.close();
      doc = inspectDoc;
      stem = assetStem;
      if (!bind(root)) { doc = null; return false; }
      Object.assign(S, { bone: -1, open: "", head: "rigged", paused: null, checking: false, allClips: false });
      paint();
      colorSkeleton();
      renderPanel();
      panel.hidden = false;
      document.body.classList.add("is-rig-inspect");
      const c = opts && opts.focus && rigChecks(doc).find((x) => x.key === opts.focus);
      if (c) {
        S.open = c.key;
        if (canShow(c)) show(c);
        else renderPanel();
      }
      return true;
    },
    close() {
      const held = mesh && (S.paused || S.head !== "rigged");
      clearGhost();
      S.head = "rigged";
      S.paused = null;
      for (const p of glbs.values()) {
        p.then((g) => g.scene.traverse((o) => {
          if (!o.isMesh) return;
          o.geometry.dispose();
          for (const m of [].concat(o.material)) { if (m.map) m.map.dispose(); m.dispose(); }
        })).catch(() => {});
      }
      glbs.clear();
      if (mesh && saved) {
        mesh.material = saved.material;
        if (saved.color) mesh.geometry.setAttribute("color", saved.color);
        else if (mesh.geometry.getAttribute("color") === colors) mesh.geometry.deleteAttribute("color");
      }
      for (const o of [helper, dots]) {
        if (!o) continue;
        scene.remove(o);
        o.geometry.dispose();
        o.material.dispose();
      }
      helper = dots = null;
      mesh = null;
      saved = null;
      doc = null;
      mouse = null;
      labels.innerHTML = "";
      tip.hidden = true;
      panel.hidden = true;
      panel.innerHTML = "";
      document.body.classList.remove("is-rig-inspect");
      if (held) resume();
    },
    // Each frame: follow the moving skeleton and keep the hover card on the point under the mouse.
    tick() {
      if (!mesh) return;
      if (!mesh.parent) { api.onClose && api.onClose(); return; }
      drawSkeleton();
      if (mouse && !down) showTip();
    },
    // A clip was played from the bar: back to the rig as it moves; the bone card shows that clip's biggest turn.
    clipChanged() {
      if (!mesh) return;
      if (S.head !== "rigged") { clearGhost(); S.head = "rigged"; }
      S.paused = null;
      renderPanel();
    },
    state: S,
    pick,
  };
  return api;
}

// Automated tests for the SteadyCam engine using synthetic camera footage.
const C = require('../src/steadycam-core.js');
const W = 160, H = 120;
let pass = 0, fail = 0;
function check(name, ok, info) { console.log((ok ? 'PASS ' : 'FAIL ') + name + (info ? '  ' + info : '')); ok ? pass++ : fail++; }

// Procedural "real-world" scene: random blobs + edges on a large canvas
const SW = 600, SH = 500, scene = new Float32Array(SW * SH);
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
for (let k = 0; k < 900; k++) {
  const bx = rnd() * SW, by = rnd() * SH, r = 2 + rnd() * 9, v = rnd() * 255 - 128, sq = rnd() < 0.4;
  for (let y = Math.max(0, by - r - 2 | 0); y < Math.min(SH, by + r + 2); y++)
    for (let x = Math.max(0, bx - r - 2 | 0); x < Math.min(SW, bx + r + 2); x++) {
      const inside = sq ? Math.abs(x - bx) < r && Math.abs(y - by) < r : (x - bx) ** 2 + (y - by) ** 2 < r * r;
      if (inside) scene[(y | 0) * SW + (x | 0)] += v * 0.5;
    }
}
for (let i = 0; i < scene.length; i++) scene[i] = Math.max(0, Math.min(255, 128 + scene[i]));
function samp(x, y) {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  if (x0 < 0 || y0 < 0 || x0 >= SW - 1 || y0 >= SH - 1) return 128;
  const i = y0 * SW + x0;
  return scene[i] * (1 - fx) * (1 - fy) + scene[i + 1] * fx * (1 - fy) + scene[i + SW] * (1 - fx) * fy + scene[i + SW + 1] * fx * fy;
}
// Render frame where scene content is transformed by similarity (ang, s, tx, ty) about frame centre
function render(ang, s, tx, ty, noise) {
  const f = new Float32Array(W * H), c = Math.cos(ang) / s, sn = Math.sin(ang) / s;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const qx = x - W / 2 - tx, qy = y - H / 2 - ty;
    const px = c * qx + sn * qy, py = -sn * qx + c * qy; // inverse rotation/scale
    f[y * W + x] = samp(px + SW / 2, py + SH / 2) + (noise ? (rnd() - 0.5) * noise : 0);
  }
  return f;
}

// --- Test 1: single-frame motion recovery (shake, roll, forward/back) ---
const cases = [
  ['translate +6,-4 px', 0, 1, 6, -4],
  ['translate -14,+9 px (big jolt)', 0, 1, -14, 9],
  ['roll 2 deg', 2 * Math.PI / 180, 1, 0, 0],
  ['move forward ~4% (zoom in)', 0, 1.04, 0, 0],
  ['move back ~4% (zoom out)', 0, 0.96, 0, 0],
  ['combined + sensor noise', 1.5 * Math.PI / 180, 1.03, 5, 3, 6],
];
for (const [name, a, s, tx, ty, noise] of cases) {
  const est = new C.MotionEstimator(W, H);
  est.process(render(0, 1, 0, 0, noise));
  const m = est.process(render(a, s, tx, ty, noise));
  const eT = Math.hypot(m.dx - tx, m.dy - ty), eA = Math.abs(m.da - a) * 180 / Math.PI, eS = Math.abs(Math.exp(m.dls) - s);
  check(name, m.ok && eT < 0.6 && eA < 0.3 && eS < 0.006,
    `est dx=${m.dx.toFixed(2)} dy=${m.dy.toFixed(2)} rot=${(m.da * 180 / Math.PI).toFixed(2)}° scale=${Math.exp(m.dls).toFixed(4)} | inliers ${est.last.inliers}/${est.last.total}`);
}

// --- Test 2: handheld sequence (slow pan + hand tremor + forward/back bob) ---
function runSequence(mode) {
  const est = new C.MotionEstimator(W, H), sm = new C.TrajectorySmoother({ mode, margin: 0.12, w: W, h: H });
  const N = 90; let rawX = [], rawY = [], rawS = [], stX = [], stY = [], stS = [];
  seed = 99;
  for (let k = 0; k < N; k++) {
    const pan = mode === 'smooth' ? 0.4 * k : 0;
    const tx = pan + 5 * Math.sin(k * 1.7) + 3 * (rnd() - 0.5);
    const ty = 4 * Math.sin(k * 2.3 + 1) + 3 * (rnd() - 0.5);
    const s = 1 + 0.03 * Math.sin(k * 0.9);               // phone moving inches front/back
    const a = 0.012 * Math.sin(k * 1.3);
    const m = est.process(render(a, s, tx, ty, 4));
    const corr = sm.update(m);
    rawX.push(tx); rawY.push(ty); rawS.push(s);
    stX.push(tx + corr.dx); stY.push(ty + corr.dy); stS.push(s * corr.scale);
  }
  const jit = (arr) => { // RMS of frame-to-frame change minus mean velocity (i.e. shake, not intentional pan)
    const d = []; for (let i = 6; i < arr.length; i++) d.push(arr[i] - arr[i - 1]);
    const mu = d.reduce((p, c) => p + c, 0) / d.length;
    return Math.sqrt(d.reduce((p, c) => p + (c - mu) ** 2, 0) / d.length);
  };
  return {
    shake: (jit(rawX) + jit(rawY)) / (jit(stX) + jit(stY)),
    zoom: jit(rawS) / jit(stS),
    rawPx: (jit(rawX) + jit(rawY)) / 2, stPx: (jit(stX) + jit(stY)) / 2,
  };
}
for (const mode of ['smooth', 'lock']) {
  const r = runSequence(mode);
  check(`${mode} mode: positional shake reduced`, r.shake > 3,
    `${r.rawPx.toFixed(2)}px -> ${r.stPx.toFixed(2)}px per frame (${((1 - 1 / r.shake) * 100).toFixed(0)}% less shake)`);
  check(`${mode} mode: front/back zoom wobble reduced (framing locked)`, r.zoom > 2,
    `${((1 - 1 / r.zoom) * 100).toFixed(0)}% less size wobble`);
}

// --- Test 3: performance budget (must run at video rate on phones) ---
const est = new C.MotionEstimator(W, H), f0 = render(0, 1, 0, 0), f1 = render(0.01, 1.01, 4, 2);
est.process(f0); const t0 = process.hrtime.bigint(); const R = 60;
for (let i = 0; i < R; i++) { est.process(i % 2 ? f0 : f1); }
const ms = Number(process.hrtime.bigint() - t0) / 1e6 / R;
check('per-frame analysis < 8 ms (desktop CPU)', ms < 8, `${ms.toFixed(2)} ms/frame`);

// --- Test 4: no false correction on a static scene ---
{ const e = new C.MotionEstimator(W, H), sm = new C.TrajectorySmoother({ w: W, h: H }); let maxC = 0;
  for (let k = 0; k < 20; k++) { const c = sm.update(e.process(render(0, 1, 0, 0, 5))); maxC = Math.max(maxC, Math.abs(c.dx), Math.abs(c.dy)); }
  check('static camera: no drift / false correction', maxC < 0.5, `max correction ${maxC.toFixed(3)}px`); }


// ================= v2: target lock-on ("homing") =================
const OW = 960, OH = 720, ZOOM = 1.25, GEO = { G: ZOOM * OW / W, W: OW, H: OH }, GEO0 = GEO;
const GEOZ = { G: 1.6 * ZOOM * OW / W, W: OW, H: OH };   // user zoomed in 1.6x
function renderSprite(a, s, tx, ty, noise, spr, hide) {
  const f = render(a, s, tx, ty, noise);
  if (hide) return f;
  const c = Math.cos(a) * s, sn = Math.sin(a) * s;
  const fx = W / 2 + tx + c * spr[0] - sn * spr[1], fy = H / 2 + ty + sn * spr[0] + c * spr[1];
  for (let y = -6; y <= 6; y++) for (let x = -6; x <= 6; x++) {
    const X = Math.round(fx + x), Y = Math.round(fy + y);
    if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
    f[Y * W + X] = (Math.abs(x) < 2 || Math.abs(y) < 2) ? 250 : ((x * y > 0) ? 20 : 200);   // drone-like marker
  }
  return f;
}
function screenOf(P, a, s, tx, ty, spr) {   // where the true target lands on the output screen
  const c = Math.cos(a) * s, sn = Math.sin(a) * s;
  return P.frameToScreen(W / 2 + tx + c * spr[0] - sn * spr[1], H / 2 + ty + sn * spr[0] + c * spr[1]);
}
function lockRun(opts) {
  const GEO = opts.geo || GEO0; seed = 4242;
  const P = new C.Pipeline(W, H, { mode: 'target' });
  let lockS = null, errs = [], rawMove = [], states = [], reacq = -1, edges = 0;
  for (let k = 0; k < 120; k++) {
    const tx = 5 * Math.sin(k * 1.9) + 3 * (rnd() - 0.5) + (opts.pan || 0) * k;
    const ty = 4 * Math.sin(k * 2.4 + 1) + 3 * (rnd() - 0.5);
    const s = 1 + 0.03 * Math.sin(k * 0.8), a = 0.015 * Math.sin(k * 1.3);
    const spr = opts.moving ? [20 + 8 * Math.sin(k * 0.05), -10 + 6 * Math.cos(k * 0.05)] : [20, -10];
    const hide = opts.hideFrom != null && k >= opts.hideFrom && k < opts.hideFrom + 15;
    P.step(renderSprite(a, s, tx, ty, 4, spr, hide), GEO, true);
    if (k === 3) {
      const c = Math.cos(a) * s, sn = Math.sin(a) * s;
      const ok = P.lockAtFrame(W / 2 + tx + c * spr[0] - sn * spr[1], H / 2 + ty + sn * spr[0] + c * spr[1]);
      if (!ok) return { fail: 'lock refused' };
      lockS = screenOf(P, a, s, tx, ty, spr);
    }
    states.push(P.targetState); if (P.atEdge) edges++;
    if (lockS && k > 6 && !hide && !(opts.hideFrom != null && k < opts.hideFrom + 25 && k >= opts.hideFrom)) {
      const q = screenOf(P, a, s, tx, ty, spr);
      errs.push(Math.hypot(q.x - lockS.x, q.y - lockS.y) / GEO.G);   // analysis px
      rawMove.push(Math.hypot(tx + 20, ty));
    }
    if (opts.hideFrom != null && k >= opts.hideFrom + 15 && reacq < 0 && P.targetState === 'locked') reacq = k - opts.hideFrom - 15;
  }
  const rms = Math.sqrt(errs.reduce((p, e) => p + e * e, 0) / errs.length);
  return { rms, max: Math.max(...errs), states, reacq, edges, G: GEO.G };
}
{ const r = lockRun({});
  check('target lock: shaking + roll + forward/back, target held on screen', !r.fail && r.rms < 0.6,
    r.fail || `target stays within ${r.rms.toFixed(2)} analysis px RMS (${(r.rms * GEO.G).toFixed(1)} px of ${OW}px output; max ${(r.max * GEO.G).toFixed(1)} px)`); }
{ const r = lockRun({ moving: true, geo: GEOZ });
  check('target lock: moving target followed (homing, 1.6x zoom)', !r.fail && r.rms < 0.6 && r.edges === 0,
    r.fail || `moving target held within ${(r.rms * r.G).toFixed(1)} px RMS on a ${OW}px output, never hit the edge`); }
{ const r = lockRun({ moving: true });
  check('target lock: out of headroom is flagged (EDGE warning shown, never fakes a lock)', r.edges > 0,
    `EDGE raised on ${r.edges} frames when movement exceeded the 10% margin`); }
{ const r = lockRun({ hideFrom: 40 });
  const lostSeen = r.states.slice(40, 56).includes('lost');
  check('target hidden 15 frames -> reports LOST, coasts, re-acquires', lostSeen && r.reacq >= 0 && r.reacq <= 5,
    `lost reported: ${lostSeen}; re-acquired ${r.reacq} frames after reappearing`); }
{ const P = new C.Pipeline(W, H, { mode: 'target' }); const flat = new Float32Array(W * H).fill(90);
  P.step(flat, GEO, true); P.step(flat, GEO, true);
  check('lock refused on a featureless spot (e.g. blank wall)', P.lockAtFrame(80, 60) === false); }
{ const P = new C.Pipeline(W, H); P.step(render(0, 1, 0, 0), GEO, true); P.step(render(0.02, 1.03, 5, -3), GEO, true);
  const s = P.frameToScreen(37.3, 81.9), f = P.screenToFrame(s.x, s.y);
  check('screen <-> frame mapping round-trip (tap-to-lock accuracy)', Math.hypot(f.x - 37.3, f.y - 81.9) < 1e-6); }
{ // headroom: output never shows black borders from translation, in any mode
  let worst = 0;
  for (const mode of ['smooth', 'lock']) { seed = 7; const P = new C.Pipeline(W, H, { mode });
    for (let k = 0; k < 60; k++) { const c = P.step(render(0, 1, 12 * Math.sin(k), 9 * Math.cos(k * 1.3), 3), GEO, true);
      const L = P._lims(GEO.G, OW, OH, c); worst = Math.max(worst, Math.abs(c.dx) - L.x, Math.abs(c.dy) - L.y); } }
  check('correction never exceeds crop headroom (no black edges)', worst <= 1e-9, `max overshoot ${Math.max(0, worst).toFixed(4)} px`); }

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

/*! SteadyCam core — software video stabilization engine (no hardware, no stand).
 *  Pipeline per frame:
 *   1. grayscale @ ~160px analysis width, 2-level pyramid
 *   2. grid feature detection (Shi-Tomasi min-eigenvalue)
 *   3. pyramidal SAD block matching + sub-pixel refinement
 *   4. RANSAC similarity fit -> dx, dy, rotation, scale (scale = forward/back motion)
 *   5. trajectory smoothing (Smooth) or hold (Tripod lock), clamped to crop margin
 *   6. caller warps the frame by the returned correction
 */
(function (root) {
  'use strict';

  function toGray(rgba, w, h, out) {
    out = out || new Float32Array(w * h);
    for (let i = 0, j = 0; i < w * h; i++, j += 4)
      out[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    return out;
  }

  function downsample2(src, w, h) {
    const w2 = w >> 1, h2 = h >> 1, out = new Float32Array(w2 * h2);
    for (let y = 0; y < h2; y++)
      for (let x = 0; x < w2; x++) {
        const i = 2 * y * w + 2 * x;
        out[y * w2 + x] = (src[i] + src[i + 1] + src[i + w] + src[i + w + 1]) * 0.25;
      }
    return { data: out, w: w2, h: h2 };
  }

  function buildPyramid(gray, w, h) {
    const l0 = { data: gray, w, h };
    return [l0, downsample2(gray, w, h)];
  }

  // Grid-based Shi-Tomasi corners: one best point per cell for even coverage.
  function detectPoints(g, w, h, gridX, gridY, border) {
    gridX = gridX || 10; gridY = gridY || 8; border = border || 12;
    const Ix = new Float32Array(w * h), Iy = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        Ix[i] = (g[i + 1] - g[i - 1]) * 0.5;
        Iy[i] = (g[i + w] - g[i - w]) * 0.5;
      }
    const cw = (w - 2 * border) / gridX, ch = (h - 2 * border) / gridY;
    const cand = [];
    let maxScore = 0;
    for (let cy = 0; cy < gridY; cy++)
      for (let cx = 0; cx < gridX; cx++) {
        let best = -1, bx = 0, by = 0;
        const x0 = Math.floor(border + cx * cw), x1 = Math.floor(border + (cx + 1) * cw);
        const y0 = Math.floor(border + cy * ch), y1 = Math.floor(border + (cy + 1) * ch);
        for (let y = y0; y < y1; y += 2)
          for (let x = x0; x < x1; x += 2) {
            let sxx = 0, syy = 0, sxy = 0;
            for (let dy = -2; dy <= 2; dy++)
              for (let dx = -2; dx <= 2; dx++) {
                const i = (y + dy) * w + x + dx, gx = Ix[i], gy = Iy[i];
                sxx += gx * gx; syy += gy * gy; sxy += gx * gy;
              }
            const tr = (sxx + syy) * 0.5, d = (sxx - syy) * 0.5;
            const s = tr - Math.sqrt(d * d + sxy * sxy);
            if (s > best) { best = s; bx = x; by = y; }
          }
        if (best > 0) { cand.push({ x: bx, y: by, s: best }); if (best > maxScore) maxScore = best; }
      }
    const thr = Math.max(1, maxScore * 0.01);
    return cand.filter(p => p.s > thr);
  }

  function sad(a, b, w, ax, ay, bx, by, r) {
    let s = 0;
    for (let dy = -r; dy <= r; dy++) {
      let ia = (ay + dy) * w + ax - r, ib = (by + dy) * w + bx - r;
      for (let dx = -r; dx <= r; dx++, ia++, ib++) s += Math.abs(a[ia] - b[ib]);
    }
    return s;
  }

  function blockSearch(A, B, ax, ay, gx, gy, R, r) {
    const w = A.w, h = A.h;
    if (ax < r || ay < r || ax >= w - r || ay >= h - r) return null;
    let best = Infinity, bx = gx, by = gy;
    for (let y = gy - R; y <= gy + R; y++) {
      if (y < r || y >= h - r) continue;
      for (let x = gx - R; x <= gx + R; x++) {
        if (x < r || x >= w - r) continue;
        const c = sad(A.data, B.data, w, ax, ay, x, y, r);
        if (c < best) { best = c; bx = x; by = y; }
      }
    }
    return best === Infinity ? null : { x: bx, y: by, c: best };
  }

  function trackPoint(PA, PB, x, y) {
    const r = 3;
    // coarse level (half res): wide search  -> handles ~16px @160w per frame
    const c = blockSearch(PA[1], PB[1], x >> 1, y >> 1, x >> 1, y >> 1, 8, 2);
    if (!c) return null;
    const f = blockSearch(PA[0], PB[0], x, y, c.x * 2 + (x & 1), c.y * 2 + (y & 1), 2, r);
    if (!f) return null;
    // sub-pixel parabola refinement
    const L = PA[0], w = L.w, h = L.h;
    let sx = 0, sy = 0;
    if (f.x > r && f.x < w - r - 1) {
      const cl = sad(L.data, PB[0].data, w, x, y, f.x - 1, f.y, r), cr = sad(L.data, PB[0].data, w, x, y, f.x + 1, f.y, r);
      const den = cl - 2 * f.c + cr; if (den > 0) sx = Math.max(-0.5, Math.min(0.5, (cl - cr) / (2 * den)));
    }
    if (f.y > r && f.y < h - r - 1) {
      const cu = sad(L.data, PB[0].data, w, x, y, f.x, f.y - 1, r), cd = sad(L.data, PB[0].data, w, x, y, f.x, f.y + 1, r);
      const den = cu - 2 * f.c + cd; if (den > 0) sy = Math.max(-0.5, Math.min(0.5, (cu - cd) / (2 * den)));
    }
    return { x: f.x + sx, y: f.y + sy };
  }

  function lsSimilarity(P, Q, idx) {
    let n = idx.length, pmx = 0, pmy = 0, qmx = 0, qmy = 0;
    for (const i of idx) { pmx += P[i].x; pmy += P[i].y; qmx += Q[i].x; qmy += Q[i].y; }
    pmx /= n; pmy /= n; qmx /= n; qmy /= n;
    let num1 = 0, num2 = 0, den = 0;
    for (const i of idx) {
      const px = P[i].x - pmx, py = P[i].y - pmy, qx = Q[i].x - qmx, qy = Q[i].y - qmy;
      num1 += px * qx + py * qy; num2 += px * qy - py * qx; den += px * px + py * py;
    }
    if (den < 1e-6) return { a: 1, b: 0, tx: qmx - pmx, ty: qmy - pmy };
    const a = num1 / den, b = num2 / den;
    return { a, b, tx: qmx - (a * pmx - b * pmy), ty: qmy - (b * pmx + a * pmy) };
  }

  function ransacSimilarity(P, Q, iters, thresh) {
    const n = P.length;
    if (n < 3) return null;
    iters = iters || 80; thresh = thresh || 1.0;
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    let bestIdx = [];
    for (let it = 0; it < iters; it++) {
      const i = Math.floor(rnd() * n), j = Math.floor(rnd() * n);
      if (i === j) continue;
      const dpx = P[j].x - P[i].x, dpy = P[j].y - P[i].y, d2 = dpx * dpx + dpy * dpy;
      if (d2 < 25) continue;
      const dqx = Q[j].x - Q[i].x, dqy = Q[j].y - Q[i].y;
      const a = (dqx * dpx + dqy * dpy) / d2, b = (dqy * dpx - dqx * dpy) / d2;
      const s = Math.hypot(a, b); if (s < 0.7 || s > 1.4) continue;
      const tx = Q[i].x - (a * P[i].x - b * P[i].y), ty = Q[i].y - (b * P[i].x + a * P[i].y);
      const idx = [];
      for (let k = 0; k < n; k++) {
        const ex = a * P[k].x - b * P[k].y + tx - Q[k].x, ey = b * P[k].x + a * P[k].y + ty - Q[k].y;
        if (ex * ex + ey * ey < thresh * thresh) idx.push(k);
      }
      if (idx.length > bestIdx.length) bestIdx = idx;
    }
    if (bestIdx.length < Math.max(3, n * 0.25)) return null;
    const m = lsSimilarity(P, Q, bestIdx);
    m.inliers = bestIdx.length; m.total = n;
    return m;
  }

  /** Estimates inter-frame camera motion from grayscale frames. */
  class MotionEstimator {
    constructor(w, h) { this.w = w; this.h = h; this.prev = null; this.last = { inliers: 0, total: 0 }; }
    reset() { this.prev = null; }
    /** returns {dx,dy,da,dls} in analysis pixels (frame-centred), or identity if unreliable */
    process(gray) {
      const pyr = buildPyramid(gray, this.w, this.h);
      const prev = this.prev; this.prev = pyr;
      const ident = { dx: 0, dy: 0, da: 0, dls: 0, ok: false };
      if (!prev) return ident;
      const pts = detectPoints(prev[0].data, this.w, this.h);
      const cx = this.w / 2, cy = this.h / 2, P = [], Q = [];
      for (const p of pts) {
        const q = trackPoint(prev, pyr, p.x, p.y);
        if (q) { P.push({ x: p.x - cx, y: p.y - cy }); Q.push({ x: q.x - cx, y: q.y - cy }); }
      }
      const m = ransacSimilarity(P, Q);
      this.last = { inliers: m ? m.inliers : 0, total: P.length };
      if (!m) return ident;
      return { dx: m.tx, dy: m.ty, da: Math.atan2(m.b, m.a), dls: Math.log(Math.hypot(m.a, m.b)), ok: true };
    }
  }

  /** Smooths the camera trajectory and returns the per-frame correction. */
  class TrajectorySmoother {
    constructor(opts) { this.set(opts || {}); this.reset(); }
    set(o) {
      this.mode = o.mode || this.mode || 'smooth';           // 'smooth' | 'lock'
      this.margin = o.margin != null ? o.margin : (this.margin != null ? this.margin : 0.1);
      this.w = o.w || this.w || 160; this.h = o.h || this.h || 120;
    }
    reset() { this.raw = [0, 0, 0, 0]; this.sm = [0, 0, 0, 0]; }
    update(m) {
      const r = this.raw, s = this.sm;
      r[0] += m.dx; r[1] += m.dy; r[2] += m.da; r[3] += m.dls;
      // smooth: low-pass (keeps intentional pans); lock: hold framing like a tripod, tiny drift-back
      const a = this.mode === 'lock' ? [0.004, 0.004, 0.01, 0.01] : [0.07, 0.07, 0.08, 0.08];
      for (let i = 0; i < 4; i++) s[i] += a[i] * (r[i] - s[i]);
      const lim = [this.margin * this.w, this.margin * this.h, 0.06, Math.log(1 + this.margin)];
      const c = [0, 0, 0, 0];
      for (let i = 0; i < 4; i++) {
        let d = s[i] - r[i];
        if (d > lim[i]) { d = lim[i]; s[i] = r[i] + d; } else if (d < -lim[i]) { d = -lim[i]; s[i] = r[i] + d; }
        c[i] = d;
      }
      return { dx: c[0], dy: c[1], da: c[2], scale: Math.exp(c[3]) };
    }
  }

  const api = { toGray, downsample2, detectPoints, trackPoint, buildPyramid, ransacSimilarity, MotionEstimator, TrajectorySmoother };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SteadyCamCore = api;
})(typeof self !== 'undefined' ? self : this);
/*! SteadyCam SDK — drop-in stabilizer for any web app that uses the camera.
 *  Usage:
 *    const cam = new SteadyCam(videoOrCanvas, { mode: 'smooth' | 'lock', crop: 0.1 });
 *    cam.start();                     // draws stabilized frames into cam.canvas
 *    const stream = cam.captureStream(30); // MediaStream for MediaRecorder / WebRTC
 */
(function (root) {
  'use strict';
  const Core = root.SteadyCamCore;

  class SteadyCam {
    constructor(source, opts) {
      opts = opts || {};
      this.source = source;
      this.canvas = opts.canvas || document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');
      this.small = document.createElement('canvas');
      this.sctx = this.small.getContext('2d', { willReadFrequently: true });
      this.aw = 160;
      this.enabled = opts.enabled !== false;
      this.mode = opts.mode || 'smooth';
      this.crop = opts.crop != null ? opts.crop : 0.1;
      this.maxWidth = opts.maxWidth || 1280;
      this.onFrame = opts.onFrame || null;
      this.smoother = new Core.TrajectorySmoother({ mode: this.mode, margin: this.crop });
      this.stats = { rawRms: 0, stabRms: 0, reduction: 0, inliers: 0, points: 0, ms: 0 };
      this._key = '';
      this._raf = 0;
      this._prevStab = [0, 0];
    }
    setMode(m) { this.mode = m; this.smoother.set({ mode: m }); }
    setCrop(c) { this.crop = c; this.smoother.set({ margin: c }); }
    setEnabled(on) { this.enabled = on; }
    reset() { this._key = ''; }
    _size() {
      const s = this.source;
      return s.videoWidth != null ? [s.videoWidth, s.videoHeight] : [s.width, s.height];
    }
    frame() {
      const [sw, sh] = this._size();
      if (!sw || !sh) return null;
      if (this.source.readyState != null && this.source.readyState < 2) return null;
      const key = sw + 'x' + sh;
      if (key !== this._key) {
        this._key = key;
        const ow = Math.min(sw, this.maxWidth);
        this.canvas.width = ow; this.canvas.height = Math.round(ow * sh / sw);
        this.ah = Math.max(48, Math.round(this.aw * sh / sw / 2) * 2);
        this.small.width = this.aw; this.small.height = this.ah;
        this.gray = new Float32Array(this.aw * this.ah);
        this.est = new Core.MotionEstimator(this.aw, this.ah);
        this.smoother.set({ w: this.aw, h: this.ah });
        this.smoother.reset();
        this._prevStab = [0, 0];
      }
      const t0 = performance.now();
      this.sctx.drawImage(this.source, 0, 0, this.aw, this.ah);
      Core.toGray(this.sctx.getImageData(0, 0, this.aw, this.ah).data, this.aw, this.ah, this.gray);
      const m = this.est.process(this.gray);
      const c = this.smoother.update(m);
      const ms = performance.now() - t0;

      // live shake meter (output pixels, exponential RMS of frame-to-frame motion)
      const k = this.canvas.width / this.aw;
      const r = this.smoother.raw;
      const sx = r[0] + c.dx, sy = r[1] + c.dy;
      const rawD = Math.hypot(m.dx, m.dy) * k;
      const stD = Math.hypot(sx - this._prevStab[0], sy - this._prevStab[1]) * k;
      this._prevStab = [sx, sy];
      const st = this.stats, a = 0.06;
      st.rawRms = Math.sqrt((1 - a) * st.rawRms ** 2 + a * rawD * rawD);
      st.stabRms = Math.sqrt((1 - a) * st.stabRms ** 2 + a * stD * stD);
      st.reduction = st.rawRms > 0.3 ? Math.max(0, 1 - st.stabRms / st.rawRms) : 0;
      st.inliers = this.est.last.inliers; st.points = this.est.last.total;
      st.ms = (1 - a) * st.ms + a * ms;
      st.rawPos = [r[0] * k, r[1] * k]; st.stabPos = [sx * k, sy * k];

      const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height, f = W / sw;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      if (this.enabled) {
        const zoom = 1 / (1 - 2 * this.crop);
        const kSrc = sw / this.aw;
        ctx.translate(W / 2, H / 2);
        ctx.scale(zoom * f, zoom * f);
        ctx.translate(c.dx * kSrc, c.dy * kSrc);
        ctx.rotate(c.da);
        ctx.scale(c.scale, c.scale);
        ctx.drawImage(this.source, -sw / 2, -sh / 2, sw, sh);
      } else {
        ctx.drawImage(this.source, 0, 0, W, H);
      }
      if (this.onFrame) this.onFrame(this.stats, c);
      return c;
    }
    start() {
      const loop = () => { this._raf = requestAnimationFrame(loop); this.frame(); };
      cancelAnimationFrame(this._raf); this._raf = requestAnimationFrame(loop);
      return this;
    }
    stop() { cancelAnimationFrame(this._raf); this._raf = 0; }
    captureStream(fps) { return this.canvas.captureStream(fps || 30); }
  }

  root.SteadyCam = SteadyCam;
})(typeof self !== 'undefined' ? self : this);

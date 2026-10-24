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
      const pyr = buildPyramid(gray.slice(), this.w, this.h);   // own copy: callers reuse their buffer
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
    /** m: frame motion; limXY (optional): exact translation headroom {x,y} in analysis px from the renderer */
    update(m, limXY) {
      const r = this.raw, s = this.sm;
      r[0] += m.dx; r[1] += m.dy; r[2] += m.da; r[3] += m.dls;
      // smooth: low-pass (keeps intentional pans); lock: hold framing like a tripod, tiny drift-back
      const a = this.mode === 'smooth' ? [0.07, 0.07, 0.08, 0.08] : [0.004, 0.004, 0.01, 0.01];
      for (let i = 0; i < 4; i++) s[i] += a[i] * (r[i] - s[i]);
      const lim = [limXY ? limXY.x : this.margin * this.w, limXY ? limXY.y : this.margin * this.h, 0.06, Math.log(1 + this.margin)];
      const c = [0, 0, 0, 0];
      for (let i = 0; i < 4; i++) {
        let d = s[i] - r[i];
        if (d > lim[i]) { d = lim[i]; s[i] = r[i] + d; } else if (d < -lim[i]) { d = -lim[i]; s[i] = r[i] + d; }
        c[i] = d;
      }
      return { dx: c[0], dy: c[1], da: c[2], scale: Math.exp(c[3]) };
    }
  }

  /**
   * Target lock-on ("homing") tracker.
   * Each frame: predict where the locked target went using the measured camera motion
   * (precise, from the whole frame) plus the target's own velocity, then confirm with a
   * zero-mean NCC template search around the prediction, and fuse with an alpha-beta filter.
   * If the target is briefly hidden it coasts on the prediction and re-acquires with a wider search.
   */
  class TargetTracker {
    constructor(w, h, opts) {
      opts = opts || {};
      this.w = w; this.h = h;
      this.r = opts.radius || Math.max(6, Math.round(w * 0.05));
      this.search = opts.search || 10;
      this.state = 'idle'; this.score = 0; this.lost = 0;
    }
    _stats(g, x, y) {
      const r = this.r, w = this.w; let s = 0, s2 = 0;
      for (let dy = -r; dy <= r; dy++) { let i = (y + dy) * w + x - r; for (let dx = -r; dx <= r; dx++, i++) { const v = g[i]; s += v; s2 += v * v; } }
      const n = (2 * r + 1) ** 2, mu = s / n;
      return [mu, Math.sqrt(Math.max(0, s2 / n - mu * mu))];
    }
    _patch(g, x, y) {
      const r = this.r, w = this.w, [mu, sd] = this._stats(g, x, y), out = new Float32Array((2 * r + 1) ** 2);
      let k = 0;
      for (let dy = -r; dy <= r; dy++) { let i = (y + dy) * w + x - r; for (let dx = -r; dx <= r; dx++, i++) out[k++] = (g[i] - mu) / (sd || 1); }
      return { p: out, sd };
    }
    _inside(x, y, pad) { const r = this.r + (pad || 0); return x >= r && y >= r && x < this.w - r && y < this.h - r; }
    _ncc(g, x, y) {
      const r = this.r, w = this.w, T = this.T, [mu, sd] = this._stats(g, x, y);
      if (sd < 1e-3) return -1;
      let s = 0, k = 0;
      for (let dy = -r; dy <= r; dy++) { let i = (y + dy) * w + x - r; for (let dx = -r; dx <= r; dx++, i++) s += T[k++] * (g[i] - mu); }
      return s / (T.length * sd);
    }
    /** Lock onto the feature at analysis pixel (x, y). Returns false on a featureless spot. */
    lock(g, x, y) {
      let ix = Math.round(x), iy = Math.round(y);
      if (!this._inside(ix, iy, 1)) return false;
      // snap to the most trackable spot (strongest 2-D texture) within a few pixels of the tap
      let bestS = -1, bx = ix, by = iy, w = this.w, r = this.r;
      for (let yy = iy - 4; yy <= iy + 4; yy++) for (let xx = ix - 4; xx <= ix + 4; xx++) {
        if (!this._inside(xx, yy, 2)) continue;
        let sxx = 0, syy = 0, sxy = 0;
        for (let dy = -r; dy <= r; dy += 2) for (let dx = -r; dx <= r; dx += 2) {
          const i = (yy + dy) * w + xx + dx, gx = (g[i + 1] - g[i - 1]) * 0.5, gy = (g[i + w] - g[i - w]) * 0.5;
          sxx += gx * gx; syy += gy * gy; sxy += gx * gy;
        }
        const tr = (sxx + syy) / 2, d = (sxx - syy) / 2, s = tr - Math.sqrt(d * d + sxy * sxy);
        const pen = s * (1 - 0.03 * Math.hypot(xx - ix, yy - iy));          // prefer the tapped spot
        if (pen > bestS) { bestS = pen; bx = xx; by = yy; }
      }
      ix = bx; iy = by;
      const { p, sd } = this._patch(g, ix, iy);
      if (sd < 3) return false;
      this.T = p; this.x = ix; this.y = iy; this.vx = 0; this.vy = 0;
      this.state = 'locked'; this.lost = 0; this.score = 1;
      return true;
    }
    unlock() { this.state = 'idle'; }
    /** g: current gray frame, m: camera motion prev->cur ({dx,dy,da,dls}, frame-centred) */
    update(g, m) {
      if (this.state === 'idle') return this;
      const cx = this.w / 2, cy = this.h / 2, s = Math.exp(m.dls || 0);
      const a = s * Math.cos(m.da || 0), b = s * Math.sin(m.da || 0);
      const px = this.x - cx, py = this.y - cy;
      const qx = a * px - b * py + (m.dx || 0) + cx + this.vx, qy = b * px + a * py + (m.dy || 0) + cy + this.vy;
      const R = this.state === 'lost' ? this.search * 2 : this.search;
      const gx = Math.round(qx), gy = Math.round(qy);
      let best = -2, bx = gx, by = gy;
      const sc = new Map();
      for (let y = gy - R; y <= gy + R; y++) for (let x = gx - R; x <= gx + R; x++) {
        if (!this._inside(x, y, 1)) continue;
        const v = this._ncc(g, x, y); sc.set(y * this.w + x, v);
        if (v > best) { best = v; bx = x; by = y; }
      }
      this.score = best;
      if (best > 0.55) {
        // sub-pixel peak
        const at = (x, y) => { const k = y * this.w + x; return sc.has(k) ? sc.get(k) : this._ncc(g, x, y); };
        let ox = 0, oy = 0;
        const l = at(bx - 1, by), rr = at(bx + 1, by), u = at(bx, by - 1), d = at(bx, by + 1);
        const dxn = l - 2 * best + rr, dyn = u - 2 * best + d;
        if (dxn < 0) ox = Math.max(-0.5, Math.min(0.5, 0.5 * (l - rr) / dxn));
        if (dyn < 0) oy = Math.max(-0.5, Math.min(0.5, 0.5 * (u - d) / dyn));
        const ix = bx + ox - qx, iy = by + oy - qy;
        // alpha-beta (homing) gains; measurement weighted by match confidence, prediction is
        // already precise because it comes from the whole-frame camera motion
        const conf = Math.min(1, Math.max(0, (best - 0.55) / 0.35));
        const G = this.state === 'lost' ? 1 : 0.3 * conf, H = 0.06 * conf;
        this.x = qx + G * ix; this.y = qy + G * iy;
        this.vx = (this.vx + H * ix) * 0.98; this.vy = (this.vy + H * iy) * 0.98;
        if (best > 0.8) {                                           // adapt to slow appearance change
          const { p } = this._patch(g, Math.round(this.x), Math.round(this.y));
          for (let k = 0; k < p.length; k++) this.T[k] = 0.95 * this.T[k] + 0.05 * p[k];
        }
        this.lost = 0; this.state = 'locked';
      } else {
        this.x = qx; this.y = qy; this.vx *= 0.9; this.vy *= 0.9;   // coast on prediction
        if (++this.lost > 10) this.state = 'lost';
      }
      if (!this._inside(this.x, this.y, 0)) { this.state = 'lost'; this.lost = Math.max(this.lost, 11); }
      return this;
    }
  }

  /**
   * Translation that puts the target (frame-centred analysis px) at the lock point, given the
   * rotation/scale correction c, clamped to the available headroom lim {x,y}.
   */
  function targetCorrection(tx, ty, lockX, lockY, c, lim) {
    const a = c.scale * Math.cos(c.da), b = c.scale * Math.sin(c.da);
    let dx = lockX - (a * tx - b * ty), dy = lockY - (b * tx + a * ty);
    const ex = Math.abs(dx) > lim.x, ey = Math.abs(dy) > lim.y;
    dx = Math.max(-lim.x, Math.min(lim.x, dx)); dy = Math.max(-lim.y, Math.min(lim.y, dy));
    return { dx, dy, atEdge: ex || ey };
  }

  /**
   * Complete per-frame stabilization logic (everything except pixel drawing).
   * Geometry: G = output pixels per analysis pixel (includes crop zoom + user zoom),
   * W,H = output size. The renderer draws: translate(W/2,H/2) scale(G) translate(c.dx,c.dy)
   * rotate(c.da) scale(c.scale) then the frame centred, in analysis units.
   */
  class Pipeline {
    constructor(aw, ah, opts) {
      opts = opts || {};
      this.aw = aw; this.ah = ah;
      this.est = new MotionEstimator(aw, ah);
      this.smoother = new TrajectorySmoother({ w: aw, h: ah, margin: 0.1 });
      this.tracker = new TargetTracker(aw, ah);
      this.setMode(opts.mode || 'smooth');
      this.c = { dx: 0, dy: 0, da: 0, scale: 1 };
      this.lockPt = null; this.atEdge = false; this.gray = null;
    }
    setMode(m) { this.mode = m; this.smoother.set({ mode: m === 'smooth' ? 'smooth' : 'lock' }); if (m !== 'target') this.unlock(); }
    get targetState() { return this.tracker.state; }
    _lims(G, W, H, c) {
      const s = c ? c.scale : 1, sa = c ? Math.abs(Math.sin(c.da)) : 0;
      return {
        x: Math.max(0, s * this.aw / 2 - W / (2 * G) - sa * s * this.ah / 2),
        y: Math.max(0, s * this.ah / 2 - H / (2 * G) - sa * s * this.aw / 2),
      };
    }
    step(gray, geom, enabled) {
      this.gray = gray; this.G = geom.G; this.W = geom.W; this.H = geom.H;
      const zoom = geom.G * this.aw / geom.W;                // total magnification vs. fit-to-output
      this.smoother.set({ margin: Math.max(0.01, (1 - 1 / Math.max(zoom, 1.0001)) / 2) });
      const m = this.est.process(gray);
      let c = this.smoother.update(m, this._lims(geom.G, geom.W, geom.H));
      this.atEdge = false;
      if (this.tracker.state !== 'idle') {
        this.tracker.update(gray, m);
        const tc = targetCorrection(this.tracker.x - this.aw / 2, this.tracker.y - this.ah / 2, this.lockPt.x, this.lockPt.y, c, this._lims(geom.G, geom.W, geom.H, c));
        c.dx = tc.dx; c.dy = tc.dy; this.atEdge = tc.atEdge;
        const r = this.smoother.raw, s = this.smoother.sm; s[0] = r[0] + c.dx; s[1] = r[1] + c.dy;   // seamless unlock
      } else {
        const L = this._lims(geom.G, geom.W, geom.H, c);
        c.dx = Math.max(-L.x, Math.min(L.x, c.dx)); c.dy = Math.max(-L.y, Math.min(L.y, c.dy));
      }
      if (!enabled) c = { dx: 0, dy: 0, da: 0, scale: 1 };
      this.c = c; this.m = m;
      return c;
    }
    /** output pixel -> analysis pixel in the current frame */
    screenToFrame(sx, sy) {
      const c = this.c, Lx = (sx - this.W / 2) / this.G - c.dx, Ly = (sy - this.H / 2) / this.G - c.dy;
      const a = Math.cos(-c.da) / c.scale, b = Math.sin(-c.da) / c.scale;
      return { x: a * Lx - b * Ly + this.aw / 2, y: b * Lx + a * Ly + this.ah / 2 };
    }
    /** analysis pixel in current frame -> output pixel */
    frameToScreen(x, y) {
      const c = this.c, a = c.scale * Math.cos(c.da), b = c.scale * Math.sin(c.da), px = x - this.aw / 2, py = y - this.ah / 2;
      return { x: this.W / 2 + this.G * (a * px - b * py + c.dx), y: this.H / 2 + this.G * (b * px + a * py + c.dy) };
    }
    /** Lock on the target at analysis pixel (x,y) of the current frame; it stays where it is on screen now. */
    lockAtFrame(x, y) {
      if (!this.gray || !this.tracker.lock(this.gray, x, y)) return false;
      const s = this.frameToScreen(this.tracker.x, this.tracker.y);
      this.lockPt = { x: (s.x - this.W / 2) / this.G, y: (s.y - this.H / 2) / this.G };
      if (this.mode !== 'target') { this.mode = 'target'; this.smoother.set({ mode: 'lock' }); }
      return true;
    }
    lockAtScreen(sx, sy) { const p = this.screenToFrame(sx, sy); return this.lockAtFrame(p.x, p.y); }
    unlock() { this.tracker.unlock(); this.lockPt = null; this.atEdge = false; }
  }

  const api = { Pipeline, toGray, downsample2, detectPoints, trackPoint, buildPyramid, ransacSimilarity, MotionEstimator, TrajectorySmoother, TargetTracker, targetCorrection };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SteadyCamCore = api;
})(typeof self !== 'undefined' ? self : this);
/*! SteadyCam SDK v2 — live, in-camera stabilization for any web app that uses the camera.
 *
 *  1) One line, every camera feature on the page gets stabilized video:
 *       SteadyCam.install({ mode: 'smooth' });       // patches getUserMedia
 *  2) Stabilize a stream you already have (recording, WebRTC, live streaming):
 *       const { stream, cam } = await SteadyCam.stabilizeStream(cameraStream);
 *  3) Full control:
 *       const cam = new SteadyCam(videoEl, { mode: 'target', output: 1080 });
 *       cam.start(); cam.lockAtScreen(x, y); cam.setZoom(2);
 *       const rec = new SteadyCam.Recorder(cam.captureStream(30), cam); rec.start(); const blob = await rec.stop();
 *
 *  Modes: 'smooth' (keeps intentional pans), 'lock' (tripod: holds framing and size),
 *         'target' (lock-on: keeps the tapped subject fixed on screen while the phone moves).
 */
(function (root) {
  'use strict';
  const Core = root.SteadyCamCore;

  class SteadyCam {
    constructor(source, opts) {
      opts = opts || {};
      this.source = source;
      this.canvas = opts.canvas || document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { alpha: false });
      this.small = document.createElement('canvas');
      this.sctx = this.small.getContext('2d', { willReadFrequently: true });
      this.aw = opts.analysisWidth || 160;
      this.enabled = opts.enabled !== false;
      this.mode = opts.mode || 'smooth';
      this.crop = opts.crop != null ? opts.crop : 0.1;          // stabilization margin per side
      this.zoom = 1;                                              // digital zoom on top
      this.output = opts.output || 1080;                          // output short side, or 'source'
      this.onFrame = opts.onFrame || null;
      this.beforeFrame = opts.beforeFrame || null;
      this.stats = { rawRms: 0, stabRms: 0, reduction: 0, inliers: 0, points: 0, ms: 0, rawPos: [0, 0], stabPos: [0, 0] };
      this._key = ''; this._raf = 0; this._lastT = -1; this._prevStab = [0, 0];
    }
    setSource(el) { this.source = el; this._key = ''; this._lastT = -1; }
    setMode(m) { this.mode = m; if (this.pipe) this.pipe.setMode(m); }
    setCrop(c) { this.crop = c; }
    setZoom(z) { this.zoom = Math.max(1, Math.min(8, z)); }
    setEnabled(on) { this.enabled = on; }
    setOutput(o) { this.output = o; this._key = ''; }
    get targetState() { return this.pipe ? this.pipe.targetState : 'idle'; }
    get atEdge() { return this.pipe ? this.pipe.atEdge : false; }
    /** Lock on whatever is at this output-canvas pixel. Returns false on a featureless spot. */
    lockAtScreen(x, y) { if (!this.pipe) return false; const ok = this.pipe.lockAtScreen(x, y); if (ok) this.mode = 'target'; return ok; }
    /** Lock on a source-video pixel. */
    lockAtSource(x, y) { if (!this.pipe) return false; const k = this.aw / this._sw; const ok = this.pipe.lockAtFrame(x * k, y * k); if (ok) this.mode = 'target'; return ok; }
    unlock() { if (this.pipe) this.pipe.unlock(); }
    /** Output pixel where the locked target sits now (for drawing a reticle), or null. */
    lockScreenPoint() { const p = this.pipe; if (!p || !p.lockPt) return null; return { x: p.W / 2 + p.G * p.lockPt.x, y: p.H / 2 + p.G * p.lockPt.y }; }
    _size() { const s = this.source; return s && s.videoWidth != null ? [s.videoWidth, s.videoHeight] : s ? [s.width, s.height] : [0, 0]; }
    _setup(sw, sh) {
      const short = Math.min(sw, sh), target = this.output === 'source' ? short : Math.min(short, this.output);
      const f = target / short;
      this.canvas.width = Math.round(sw * f / 2) * 2; this.canvas.height = Math.round(sh * f / 2) * 2;
      this.ah = Math.max(48, Math.round(this.aw * sh / sw / 2) * 2);
      this.small.width = this.aw; this.small.height = this.ah;
      this.gray = new Float32Array(this.aw * this.ah);
      const keep = this.pipe && this.pipe.mode;
      this.pipe = new Core.Pipeline(this.aw, this.ah, { mode: this.mode === 'target' ? 'target' : this.mode });
      this._sw = sw; this._sh = sh; this._prevStab = [0, 0];
      return keep;
    }
    frame() {
      if (this.beforeFrame) this.beforeFrame();
      const [sw, sh] = this._size();
      if (!sw || !sh) return null;
      const s = this.source;
      if (s.readyState != null) {
        if (s.readyState < 2) return null;
        if (s.currentTime === this._lastT && !s.srcObject) return null;   // file video paused / same frame
        this._lastT = s.currentTime;
      }
      const key = sw + 'x' + sh + ':' + this.output;
      if (key !== this._key) { this._key = key; this._setup(sw, sh); }
      const W = this.canvas.width, H = this.canvas.height;
      const t0 = performance.now();
      this.sctx.drawImage(s, 0, 0, this.aw, this.ah);
      Core.toGray(this.sctx.getImageData(0, 0, this.aw, this.ah).data, this.aw, this.ah, this.gray);
      const cropZoom = this.enabled ? 1 / (1 - 2 * this.crop) : 1;
      const G = cropZoom * this.zoom * W / this.aw;
      if (this.pipe.mode !== this.mode) this.pipe.setMode(this.mode);   // 'target' with nothing locked behaves as tripod
      const c = this.pipe.step(this.gray, { G, W, H }, this.enabled);
      const m = this.pipe.m;
      const ms = performance.now() - t0;

      // live shake meter in output pixels
      const k = G / cropZoom / this.zoom, r = this.pipe.smoother.raw;
      const sx = r[0] + c.dx, sy = r[1] + c.dy;
      const rawD = Math.hypot(m.dx, m.dy) * k, stD = Math.hypot(sx - this._prevStab[0], sy - this._prevStab[1]) * k;
      this._prevStab = [sx, sy];
      const st = this.stats, a = 0.06;
      st.rawRms = Math.sqrt((1 - a) * st.rawRms ** 2 + a * rawD * rawD);
      st.stabRms = Math.sqrt((1 - a) * st.stabRms ** 2 + a * stD * stD);
      st.reduction = st.rawRms > 0.3 ? Math.max(0, 1 - st.stabRms / st.rawRms) : 0;
      st.inliers = this.pipe.est.last.inliers; st.points = this.pipe.est.last.total;
      st.ms = (1 - a) * st.ms + a * ms; st.rawPos = [r[0] * k, r[1] * k]; st.stabPos = [sx * k, sy * k];
      st.target = this.pipe.targetState; st.edge = this.pipe.atEdge; st.score = this.pipe.tracker.score;

      // one resample, straight from the full-resolution camera frame
      const ctx = this.ctx, kSrc = sw / this.aw;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      ctx.translate(W / 2, H / 2);
      ctx.scale(G / kSrc, G / kSrc);
      ctx.translate(c.dx * kSrc, c.dy * kSrc);
      ctx.rotate(c.da);
      ctx.scale(c.scale, c.scale);
      ctx.drawImage(s, -sw / 2, -sh / 2, sw, sh);
      if (this.onFrame) this.onFrame(st, c);
      return c;
    }
    start() {
      cancelAnimationFrame(this._raf);
      const loop = () => { this._raf = requestAnimationFrame(loop); try { this.frame(); } catch (e) { console.error('SteadyCam frame', e); } };
      this._raf = requestAnimationFrame(loop);
      return this;
    }
    stop() { cancelAnimationFrame(this._raf); this._raf = 0; }
    captureStream(fps) { return this.canvas.captureStream(fps || 30); }

    /** Wrap a camera MediaStream; returns a stabilized MediaStream (audio passed through untouched). */
    static async stabilizeStream(input, opts) {
      opts = opts || {};
      const v = document.createElement('video');
      v.muted = true; v.playsInline = true; v.setAttribute('playsinline', ''); v.srcObject = input;
      Object.assign(v.style, { position: 'fixed', width: '2px', height: '2px', opacity: '0.01', pointerEvents: 'none', left: '0', top: '0' });
      document.body.appendChild(v);                                // iOS only decodes attached videos reliably
      await v.play().catch(() => {});
      const vt = input.getVideoTracks()[0], fps = (vt && vt.getSettings().frameRate) || opts.fps || 30;
      const cam = new SteadyCam(v, opts).start();
      const out = cam.captureStream(Math.round(fps));
      input.getAudioTracks().forEach(t => out.addTrack(t));
      const ot = out.getVideoTracks()[0], origStop = ot.stop.bind(ot);
      ot.stop = () => { origStop(); cam.stop(); input.getTracks().forEach(t => t.stop()); v.remove(); };
      ot.steadycam = cam; ot.cameraTrack = vt;                     // zoom/torch stay reachable via cameraTrack
      return { stream: out, cam, cameraTrack: vt };
    }

    /** One-line plugin: every getUserMedia video stream on this page comes back stabilized. */
    static install(opts) {
      const md = navigator.mediaDevices;
      if (!md || md.__steadycam) return;
      const orig = md.getUserMedia.bind(md);
      md.getUserMedia = async (constraints) => {
        const s = await orig(constraints);
        if (!constraints || !constraints.video || (opts && opts.enabled === false)) return s;
        return (await SteadyCam.stabilizeStream(s, opts)).stream;
      };
      md.__steadycam = true;
    }
  }

  /** Recorder: high-bitrate MP4 (Safari) or WebM, resolves a Blob on stop. */
  class Recorder {
    constructor(stream, cam, opts) {
      opts = opts || {};
      this.stream = stream;
      const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm'];
      this.type = types.find(t => root.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
      const px = cam ? cam.canvas.width * cam.canvas.height : 1920 * 1080;
      this.bitrate = opts.bitrate || Math.min(50e6, Math.max(8e6, px * 30 * 0.2));   // ~12 Mbps @1080p30
    }
    get extension() { return (this.rec && this.rec.mimeType || this.type).includes('mp4') ? 'mp4' : 'webm'; }
    start() {
      this.chunks = [];
      const o = { videoBitsPerSecond: this.bitrate, audioBitsPerSecond: 160000 };
      if (this.type) o.mimeType = this.type;
      this.rec = new MediaRecorder(this.stream, o);
      this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
      this.done = new Promise(res => { this.rec.onstop = () => res(new Blob(this.chunks, { type: this.rec.mimeType || this.type || 'video/webm' })); });
      this.rec.start(500);
      this.startedAt = Date.now();
    }
    get recording() { return !!this.rec && this.rec.state === 'recording'; }
    stop() { if (this.rec && this.rec.state !== 'inactive') this.rec.stop(); return this.done; }
  }

  SteadyCam.Recorder = Recorder;
  SteadyCam.version = '2.0.0';
  root.SteadyCam = SteadyCam;
})(typeof self !== 'undefined' ? self : this);

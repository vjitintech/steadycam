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

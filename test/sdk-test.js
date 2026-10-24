// End-to-end test of the browser SDK (steadycam-dom.js) with a stubbed canvas:
// checks the exact pixel transform the SDK draws with, tap-to-lock, zoom, output quality and install().
global.self = global;
global.performance = global.performance || require('perf_hooks').performance;
global.requestAnimationFrame = () => 0; global.cancelAnimationFrame = () => {};
let current = null;           // current synthetic frame {gray, w, h}
class M { constructor(){ this.a=1;this.b=0;this.c=0;this.d=1;this.e=0;this.f=0; }
  mul(a,b,c,d,e,f){ const A=this; const n={a:A.a*a+A.c*b,b:A.b*a+A.d*b,c:A.a*c+A.c*d,d:A.b*c+A.d*d,e:A.a*e+A.c*f+A.e,f:A.b*e+A.d*f+A.f}; Object.assign(this,n); }
  apply(x,y){ return [this.a*x+this.c*y+this.e, this.b*x+this.d*y+this.f]; } }
function ctxStub(canvas){ const m=new M(); return { canvas, m,
  setTransform(a,b,c,d,e,f){ Object.assign(m,{a,b,c,d,e,f}); }, translate(x,y){ m.mul(1,0,0,1,x,y); }, scale(x,y){ m.mul(x,0,0,y,0,0); },
  rotate(t){ m.mul(Math.cos(t),Math.sin(t),-Math.sin(t),Math.cos(t),0,0); }, fillRect(){}, drawImage(){ this.lastDraw = Object.assign(new M(), m); },
  getImageData(x,y,w,h){ const d=new Uint8ClampedArray(w*h*4); const g=current.gray;
    for(let yy=0;yy<h;yy++)for(let xx=0;xx<w;xx++){ const v=g[Math.min(current.h-1,Math.floor(yy*current.h/h))*current.w+Math.min(current.w-1,Math.floor(xx*current.w/w))]; const i=(yy*w+xx)*4; d[i]=d[i+1]=d[i+2]=v; d[i+3]=255; }
    return {data:d}; } }; }
global.document = { createElement: (t) => { const c = { width: 0, height: 0, style: {} }; c.getContext = () => (c._ctx = c._ctx || ctxStub(c)); c.captureStream = () => ({ getVideoTracks: () => [], addTrack(){} }); return c; } };
global.SteadyCamCore = require('../src/steadycam-core.js'); require('../src/steadycam-dom.js');
const SC = global.SteadyCam;
let pass = 0, fail = 0; const check = (n, ok, i) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (i ? '  ' + i : '')); ok ? pass++ : fail++; };

// synthetic 1280x720 "camera": textured scene at 160x90 analysis res, upsampled 8x by the stub
const AW = 160, AH = 90; let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const SW = 500, SH = 400, scene = new Float32Array(SW * SH).fill(128);
for (let k = 0; k < 700; k++) { const bx = rnd()*SW, by = rnd()*SH, r = 2+rnd()*7, v = rnd()*200-100;
  for (let y = Math.max(0,by-r|0); y < Math.min(SH,by+r); y++) for (let x = Math.max(0,bx-r|0); x < Math.min(SW,bx+r); x++) if ((x-bx)**2+(y-by)**2 < r*r) scene[(y|0)*SW+(x|0)] = Math.max(0,Math.min(255,scene[(y|0)*SW+(x|0)]+v)); }
const samp = (x,y) => { const x0=Math.floor(x),y0=Math.floor(y),fx=x-x0,fy=y-y0; if(x0<0||y0<0||x0>=SW-1||y0>=SH-1) return 128; const i=y0*SW+x0;
  return scene[i]*(1-fx)*(1-fy)+scene[i+1]*fx*(1-fy)+scene[i+SW]*(1-fx)*fy+scene[i+SW+1]*fx*fy; };
function frame(a, s, tx, ty) { const g = new Float32Array(AW*AH), c = Math.cos(a)/s, sn = Math.sin(a)/s;
  for (let y=0;y<AH;y++) for (let x=0;x<AW;x++){ const qx=x-AW/2-tx, qy=y-AH/2-ty; g[y*AW+x]=samp(c*qx+sn*qy+SW/2, -sn*qx+c*qy+SH/2); }
  current = { gray: g, w: AW, h: AH }; }
const video = { videoWidth: 1280, videoHeight: 720, readyState: 4, currentTime: 0, srcObject: {} };
const k = 1280 / AW;   // source px per analysis px
const truthSrc = (a, s, tx, ty, p) => { const c=Math.cos(a)*s, sn=Math.sin(a)*s; return [(AW/2+tx+c*p[0]-sn*p[1])*k, (AH/2+ty+sn*p[0]+c*p[1])*k]; };

function run(opts) {
  seed = 5; const cam = new SC(video, Object.assign({ mode: 'target', output: 720 }, opts.cam || {}));
  if (opts.zoom) cam.setZoom(opts.zoom);
  const P = [25, 8]; let lockScreen = null, errs = [], drawn = null;
  for (let i = 0; i < 90; i++) {
    const tx = 4*Math.sin(i*1.9)+2*(rnd()-.5), ty = 3*Math.sin(i*2.4+1)+2*(rnd()-.5), s = 1+0.025*Math.sin(i*0.8), a = 0.012*Math.sin(i*1.3);
    frame(a, s, tx, ty); video.currentTime += 1/30; cam.frame();
    const T = truthSrc(a, s, tx, ty, P), m = cam.ctx.lastDraw;
    // drawImage draws the source centred: source pixel (u,v) lands at m.apply(u - sw/2, v - sh/2)
    const scr = m.apply(T[0] - 640, T[1] - 360);
    if (i === 4) { if (opts.tap) { const ok = cam.lockAtScreen(scr[0], scr[1]); if (!ok) return { fail: true }; } else cam.lockAtSource(T[0], T[1]); lockScreen = { x: scr[0], y: scr[1] }; }
    if (lockScreen && i > 8) errs.push(Math.hypot(scr[0] - lockScreen.x, scr[1] - lockScreen.y));
    drawn = { W: cam.canvas.width, H: cam.canvas.height };
  }
  return { rms: Math.sqrt(errs.reduce((p, e) => p + e*e, 0) / errs.length), drawn, cam };
}
{ const r = run({}); check('SDK draw transform keeps locked target on its screen spot (lockAtSource)', r.rms < 4, `${r.rms.toFixed(2)} px RMS on ${r.drawn.W}x${r.drawn.H} output`); }
{ const r = run({ tap: true }); check('SDK tap-to-lock (lockAtScreen) holds the tapped subject', !r.fail && r.rms < 4, r.fail ? 'lock refused' : `${r.rms.toFixed(2)} px RMS`); }
{ const r = run({ zoom: 2 }); check('SDK with 2x digital zoom still holds the lock', r.rms < 6, `${r.rms.toFixed(2)} px RMS`); }
{ const cam = new SC(video, { output: 1080 }); frame(0,1,0,0); video.currentTime += 1; cam.frame();
  check('Output quality: 720p camera source is never upscaled past its own detail', cam.canvas.width === 1280 && cam.canvas.height === 720, `${cam.canvas.width}x${cam.canvas.height}`);
  const v4k = Object.assign({}, video, { videoWidth: 3840, videoHeight: 2160 }); const c2 = new SC(v4k, { output: 1080 }); c2.frame();
  check('Output quality: 4K capture -> clean 1080p output', c2.canvas.width === 1920 && c2.canvas.height === 1080, `${c2.canvas.width}x${c2.canvas.height}`);
  const vp = Object.assign({}, video, { videoWidth: 2160, videoHeight: 3840 }); const c3 = new SC(vp, { output: 1080 }); c3.frame();
  check('Portrait phone video keeps portrait output', c3.canvas.width === 1080 && c3.canvas.height === 1920, `${c3.canvas.width}x${c3.canvas.height}`); }
{ const r = new SC.Recorder({}, { canvas: { width: 1920, height: 1080 } }); check('Recorder bitrate scales with quality (1080p30 ~12 Mbps)', Math.abs(r.bitrate - 12.44e6) < 1e5, (r.bitrate/1e6).toFixed(1) + ' Mbps'); }
(async () => {
  const audioOnly = { getTracks: () => [], getAudioTracks: () => [], getVideoTracks: () => [] };
  Object.defineProperty(global, 'navigator', { value: { mediaDevices: { getUserMedia: async () => audioOnly } }, configurable: true, writable: true });
  SC.install(); const g1 = navigator.mediaDevices.getUserMedia; SC.install();
  const same = await navigator.mediaDevices.getUserMedia({ audio: true });
  check('install(): patches getUserMedia once; audio-only requests pass through untouched', navigator.mediaDevices.__steadycam === true && same === audioOnly && g1 === navigator.mediaDevices.getUserMedia);
  console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail ? 1 : 0);
})();

(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.append(e); return e; };
  const T = (parent, x, y, text, cls, anchor = 'middle') => { const e = S('text', { x, y, class: cls, 'text-anchor': anchor }, parent); e.textContent = text; return e; };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (v) => (Math.abs(v) < 0.005 ? '0.00' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)).replace('-', '−');
  const sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  const pn = (a) => { let p = 0, n = 0; for (const v of a) { if (v > 0) p += v; else n -= v; } return [p, n]; };
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const slope = (x) => 0.5 * (1 + erf(x / Math.SQRT2)) + x * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  const RULES = { add: 'split', ln: 'rule', gelu: 'rule', ls: 'rule' }, ZLO = -3, ZHI = 3, NB = 24;

  const model = (d) => {
    const C = d.channels, G = d.grid, N = G * G, H = 4 * C;
    const norm = (x, gam) => {
      let m = 0; for (const v of x) m += v; m /= x.length;
      let s = 0; for (const v of x) s += (v - m) ** 2;
      const sd = Math.sqrt(s / x.length + 1e-5);
      return { m, sd, n: x.map((v, i) => gam[i] * (v - m) / sd) };
    };
    const zplus = (a, W, R, I, O) => {
      const out = new Float64Array(I);
      for (let o = 0; o < O; o++) {
        if (!R[o]) continue;
        let z = 0; for (let i = 0; i < I; i++) { const c = a[i] * W[o * I + i]; if (c > 0) z += c; }
        if (z <= 0) continue;
        for (let i = 0; i < I; i++) { const c = a[i] * W[o * I + i]; if (c > 0) out[i] += c * R[o] / z; }
      }
      return out;
    };
    const normBack = (x, gam, Rn, mode) => {
      const n = x.length, { m, sd } = norm(x, gam), out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (!Rn[i]) continue;
        if (mode === 'none') { const xi = (x[i] - m) / sd; for (let j = 0; j < n; j++) out[j] += gam[i] / sd * ((i === j) - 1 / n - xi * (x[j] - m) / sd / n) * Rn[i]; continue; }
        const ni = gam[i] * (x[i] - m) / sd;
        if (Math.abs(ni) < 1e-12) continue;
        for (let j = 0; j < n; j++) out[j] += gam[i] / sd * ((i === j) - 1 / n) * x[j] * Rn[i] / ni;
      }
      return out;
    };
    const col = (a, p) => Float64Array.from({ length: C }, (_, c) => a[c * N + p]);

    const forward = (pixels) => {
      const x = Float64Array.from(pixels, (v) => v / 255);
      let h = new Float64Array(C * N);
      for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) {
        const y = (p / G) | 0, xx = p % G;
        for (let k = 0; k < 4; k++) h[c * N + p] += d.stem[c * 4 + k] * x[(2 * y + (k >> 1)) * 28 + 2 * xx + (k & 1)];
      }
      const stem = h, blocks = [];
      for (const b of d.blocks) {
        const dd = new Float64Array(C * N);
        for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) {
          const y = (p / G) | 0, xx = p % G; let s = 0;
          for (let ky = 0; ky < 7; ky++) { const iy = y + ky - 3; if (iy < 0 || iy >= G) continue;
            for (let kx = 0; kx < 7; kx++) { const ix = xx + kx - 3; if (ix < 0 || ix >= G) continue; s += b.dw[c * 49 + ky * 7 + kx] * h[c * N + iy * G + ix]; } }
          dd[c * N + p] = s;
        }
        const n = [], z = [], g = [], v = new Float64Array(C * N);
        for (let p = 0; p < N; p++) {
          const np = norm(col(dd, p), b.ln).n, zp = new Float64Array(H), gp = new Float64Array(H);
          for (let o = 0; o < H; o++) { let s = 0; for (let i = 0; i < C; i++) s += b.f1[o * C + i] * np[i]; zp[o] = s; gp[o] = gelu(s); }
          for (let c = 0; c < C; c++) { let s = 0; for (let o = 0; o < H; o++) s += b.f2[c * H + o] * gp[o]; v[c * N + p] = b.gamma[c] * s; }
          n.push(np); z.push(zp); g.push(gp);
        }
        const out = Float64Array.from(h, (val, i) => val + v[i]);
        blocks.push({ x: h, d: dd, n, z, g, v, out });
        h = out;
      }
      const P = Float64Array.from({ length: C }, (_, c) => { let s = 0; for (let p = 0; p < N; p++) s += h[c * N + p]; return s / N; });
      const q = norm(P, d.norm).n;
      const logits = Float64Array.from({ length: 10 }, (_, k) => { let s = 0; for (let c = 0; c < C; c++) s += d.head[k * C + c] * q[c]; return s; });
      return { x, stem, blocks, h, P, q, logits };
    };

    const explain = (f, k, cfg) => {
      const R = new Float64Array(10); R[k] = f.logits[k] < 0 ? -1 : 1;
      const Rq = zplus(f.q, d.head, R, C, 10), Rp = normBack(f.P, d.norm, Rq, cfg.ln);
      let Rh = new Float64Array(C * N);
      for (let c = 0; c < C; c++) {
        let s = 0; for (let p = 0; p < N; p++) s += Math.max(0, f.h[c * N + p]);
        if (s > 0) for (let p = 0; p < N; p++) Rh[c * N + p] = Math.max(0, f.h[c * N + p]) / s * Rp[c];
      }
      const maps = [Rh], flows = [];
      for (let bi = d.blocks.length - 1; bi >= 0; bi--) {
        const b = d.blocks[bi], st = f.blocks[bi], Rx = new Float64Array(C * N), Rv = new Float64Array(C * N);
        for (let i = 0; i < C * N; i++) {
          if (cfg.add === 'copy') { Rx[i] = Rh[i]; Rv[i] = Rh[i]; continue; }
          if (Math.abs(st.out[i]) > 1e-12) { Rx[i] = st.x[i] / st.out[i] * Rh[i]; Rv[i] = st.v[i] / st.out[i] * Rh[i]; }
        }
        const Rd = new Float64Array(C * N), t = { in: sum(Rv), skip: sum(Rx), ls: 0, f2: 0, gelu: 0, f1: 0, ln: 0, dw: 0 };
        const vec = () => new Float64Array(C), split = [0, 0];
        for (let i = 0; i < C * N; i++) if (Math.abs(st.out[i]) > 1e-12) { split[0] += st.x[i] / st.out[i] * Rh[i]; split[1] += st.v[i] / st.out[i] * Rh[i]; }
        const card = { add: { in: sum(Rh), split, copy: [sum(Rh), sum(Rh)] }, ln: { in: 0, rule: vec(), none: vec() }, gelu: { in: 0, rule: new Float64Array(NB), none: new Float64Array(NB) }, ls: { in: t.in, rule: vec(), none: vec() } };
        for (let p = 0; p < N; p++) {
          for (let c = 0; c < C; c++) { card.ls.rule[c] += Rv[c * N + p]; card.ls.none[c] += b.gamma[c] * Rv[c * N + p]; }
          const Ru = Float64Array.from({ length: C }, (_, c) => (cfg.ls === 'rule' ? 1 : b.gamma[c]) * Rv[c * N + p]);
          const Rg = zplus(st.g[p], b.f2, Ru, H, C);
          for (let o = 0; o < H; o++) { const z = st.z[p][o], bin = clamp(Math.floor((z - ZLO) / (ZHI - ZLO) * NB), 0, NB - 1); card.gelu.in += Rg[o]; card.gelu.rule[bin] += Rg[o]; card.gelu.none[bin] += slope(z) * Rg[o]; }
          const Rz = cfg.gelu === 'rule' ? Rg : Rg.map((r, o) => slope(st.z[p][o]) * r);
          const Rn = zplus(st.n[p], b.f1, Rz, C, H);
          const Rdp = normBack(col(st.d, p), b.ln, Rn, cfg.ln), other = normBack(col(st.d, p), b.ln, Rn, cfg.ln === 'rule' ? 'none' : 'rule');
          t.ls += sum(Ru); t.f2 += sum(Rg); t.gelu += sum(Rz); t.f1 += sum(Rn); t.ln += sum(Rdp);
          card.ln.in += sum(Rn);
          for (let c = 0; c < C; c++) { Rd[c * N + p] = Rdp[c]; card.ln[cfg.ln][c] += Rdp[c]; card.ln[cfg.ln === 'rule' ? 'none' : 'rule'][c] += other[c]; }
        }
        const Rw = new Float64Array(C * N);
        for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) {
          const r = Rd[c * N + p]; if (!r) continue;
          const y = (p / G) | 0, xx = p % G; let z = 0;
          for (let pass = 0; pass < 2; pass++) for (let ky = 0; ky < 7; ky++) { const iy = y + ky - 3; if (iy < 0 || iy >= G) continue;
            for (let kx = 0; kx < 7; kx++) { const ix = xx + kx - 3; if (ix < 0 || ix >= G) continue;
              const q = c * N + iy * G + ix, cc = st.x[q] * b.dw[c * 49 + ky * 7 + kx]; if (cc <= 0) continue;
              if (pass === 0) z += cc; else Rw[q] += cc * r / z; } }
        }
        t.dw = sum(Rw); t.bands = { skip: pn(Rx), in: pn(Rv), out: pn(Rw) };
        t.card = card;
        Rh = Float64Array.from(Rx, (v, i) => v + Rw[i]);
        flows[bi] = t;
        maps.push(Rh);
      }
      const Rpix = new Float64Array(784);
      for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) {
        const r = Rh[c * N + p]; if (!r) continue;
        const y = (p / G) | 0, xx = p % G, idx = [0, 1, 2, 3].map((k) => (2 * y + (k >> 1)) * 28 + 2 * xx + (k & 1));
        let z = 0; idx.forEach((q, k) => { const cc = f.x[q] * d.stem[c * 4 + k]; if (cc > 0) z += cc; });
        if (z > 0) idx.forEach((q, k) => { const cc = f.x[q] * d.stem[c * 4 + k]; if (cc > 0) Rpix[q] += cc * r / z; });
      }
      maps.push(Rpix);
      return { maps: maps.reverse(), flows, start: R[k] };
    };
    return { C, G, N, H, forward, explain };
  };

  const ramp = (stops) => (t) => { const x = clamp(t, 0, 1) * 3, i = Math.min(2, Math.floor(x)), f = x - i; return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f)); };
  const red = ramp([[255, 255, 255], [240, 170, 160], [200, 65, 59], [110, 16, 14]]);
  const blue = ramp([[255, 255, 255], [172, 196, 238], [52, 108, 196], [16, 40, 104]]);
  const grey = (t) => { const v = Math.round(255 - 205 * clamp(t, 0, 1)); return [v, v, v + 2]; };
  const cv = document.createElement('canvas');
  const image = (vals, n, colour) => {
    cv.width = cv.height = n; const ctx = cv.getContext('2d'), im = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) im.data.set([...colour(vals[i]), 255], i * 4);
    ctx.putImageData(im, 0, 0); return cv.toDataURL();
  };
  const signed = (vals, n) => { let m = 1e-12; for (const v of vals) m = Math.max(m, Math.abs(v)); return image(vals, n, (v) => (v >= 0 ? red(v / m) : blue(-v / m))); };

  const W = 1400, HGT = 486, R1 = 132, R2 = 404, MS = 76, GAP = 6, ARC = 52;
  const XS = [230, 470, 710, 950, 1190], XSTEM = 146, XHEAD = 1282, XCLS = 1368;
  const VW = 500, VTOP = 84, VH = W + VTOP;
  const narrow = matchMedia('(max-width: 640px)');
  const upright = (e, cx, cy) => e.setAttribute('transform', `matrix(0 1 1 0 ${cx - cy} ${cy - cx})`);

  const figure = async (root) => {
    const svg = root.querySelector('.lrpc-scene'), pad = root.querySelector('.lrpw-pad'), clearBtn = root.querySelector('.lrpw-clear'), readoutEl = root.querySelector('.lrpw-readout');
    let data;
    try { data = await (await fetch(root.dataset.src)).json(); } catch (e) { readoutEl.textContent = 'The network could not be loaded.'; return; }
    const M = model(data), B = data.blocks.length;
    const state = { pixels: data.start, cfg: { ...RULES }, look: null };
    const cardsEl = root.querySelector('.lrpc-cards'), summaryEl = root.querySelector('.lrpc-more summary'), panelText = root.querySelector('.lrpc-inside-text');
    let f, pred, ex;
    const cache = new Map();

    const arrow = (g, d, cls) => S('path', { d, class: cls, 'marker-end': 'url(#lrpc-head)' }, g);
    let scale = 16;
    const width = (a) => (scale * a < 0.7 ? 0 : scale * a);
    const band = (g, shape) => ({ shape, pos: S('path', { class: 'lrpc-band' }, g), neg: S('path', { class: 'lrpc-band lrpc-band--neg' }, g) });
    const setBand = (b, [P, N]) => {
      const wp = width(P), wn = width(N);
      b.pos.setAttribute('d', b.shape(-1, wp)); b.pos.style.strokeWidth = wp; b.pos.style.display = wp ? '' : 'none';
      b.neg.setAttribute('d', b.shape(1, wn)); b.neg.style.strokeWidth = wn; b.neg.style.display = wn ? '' : 'none';
    };
    const scene = {};
    const build = () => {
      svg.replaceChildren();
      const tall = narrow.matches, IN = tall ? 120 : 88, XIN = tall ? 36 : 52;
      svg.setAttribute('viewBox', tall ? `0 ${-VTOP} ${VW} ${VH}` : `0 0 ${W} ${HGT}`);
      const defs = S('defs', {}, svg), mk = S('marker', { id: 'lrpc-head', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M0,0 L10,5 L0,10 z', class: 'lrpc-arrowhead' }, mk);
      const sc = tall ? S('g', { transform: 'matrix(0 1 1 0 0 0)' }, svg) : svg;
      if (tall) { T(sc, -58, R1, 'Prediction', 'lrpc-row'); T(sc, -58, R2, 'Explanation', 'lrpc-row lrpc-row--rel'); }
      else { T(sc, 8, 30, 'Prediction', 'lrpc-row', 'start'); T(sc, 8, R2 - 98, 'Explanation', 'lrpc-row lrpc-row--rel', 'start'); }
      scene.rows = [R1, R2].map((y, r) => {
        const g = S('g', {}, sc), rel = r === 1, out = { maps: [], boxes: [], totals: [], parts: [], skips: [], bands: { blocks: [] } };
        const gB = S('g', {}, g), lane = (sign, w) => y + sign * (GAP / 2 + w / 2), hz = (a, b) => (sign, w) => `M${a},${lane(sign, w)} L${b},${lane(sign, w)}`;
        const sq = (x, size, label) => {
          const img = S('image', { x: x - size / 2, y: y - size / 2, width: size, height: size, preserveAspectRatio: 'none', style: 'image-rendering: pixelated' }, g);
          S('rect', { class: 'lrpc-frame', x: x - size / 2, y: y - size / 2, width: size, height: size }, g);
          if (label && !tall) T(g, x, y - size / 2 - 8, label, 'lrpc-label');
          else if (label && !rel) T(g, size === IN ? x - size / 2 - 8 : x + 4, size === IN ? y : y - size / 2 - 8, label, 'lrpc-label', size === IN ? 'middle' : 'end');
          else if (label && size === IN) T(g, x + 4, y + size / 2 + 8, label, 'lrpc-label', 'start');
          return img;
        };
        const box = (x, w, lines, cls = '') => {
          const bg = S('g', { class: `lrpc-box ${cls}` }, g);
          S('rect', { x: x - w / 2, y: y - (tall ? w : 48) / 2, width: w, height: tall ? w : 48, rx: 8 }, bg);
          const texts = lines.map((t, i) => T(bg, x, y - (lines.length - 1) * 7 + i * 14 + 4, t, i ? 'lrpc-boxsub' : 'lrpc-boxtitle'));
          return { g: bg, texts };
        };
        const dir = (a, b) => (rel ? null : arrow(g, `M${a},${y} L${b},${y}`, 'lrpc-wire'));
        out.input = sq(XIN, IN, rel ? 'pixels' : 'input');
        dir(XIN + IN / 2 + 3, XSTEM - 39);
        box(XSTEM, 72, ['stem', '2×2 patches']);
        dir(XSTEM + 39, XS[0] - MS / 2 - 3);
        if (rel) { out.bands.pix = band(gB, hz(XIN + IN / 2, XSTEM - 36)); out.bands.stem = band(gB, hz(XSTEM + 36, XS[0] - MS / 2)); out.bands.top = band(gB, hz(XS[B] + MS / 2, XHEAD - 35)); }
        XS.forEach((x, k) => {
          out.maps.push(sq(x, MS, k ? `after block ${k}` : 'after the stem'));
          out.totals.push(!rel ? null : tall ? T(g, x - 4, y - MS / 2 - 8, '', 'lrpc-total', 'end') : T(g, x - MS / 2, y + MS / 2 + 18, '', 'lrpc-total', 'start'));
          out.parts.push(!rel ? null : tall ? T(g, x + 12, y - MS / 2 - 8, '', 'lrpc-parts', 'end') : T(g, x - MS / 2, y + MS / 2 + 32, '', 'lrpc-parts', 'start'));
          if (k === B) return;
          const L = x + MS / 2 + 20, bx = x + 114, px = x + 172, nx = XS[k + 1];
          dir(x + MS / 2 + 3, bx - 39);
          out.boxes.push(box(bx, 70, rel ? [`block ${k + 1}`, '', ''] : [`block ${k + 1}`], 'lrpc-block'));
          dir(bx + 39, px - 14);
          if (!rel) { S('circle', { class: 'lrpc-plus', cx: px, cy: y, r: 11 }, g); T(g, px, y, '+', 'lrpc-plussign').setAttribute('dominant-baseline', 'central'); }
          if (rel) S('circle', { cx: px, cy: y, r: 15, fill: 'transparent', style: 'cursor: pointer' }, g).addEventListener('click', () => look(k + 1));
          dir(px + 14, nx - MS / 2 - 3);
          if (rel) {
            const pair = () => ({ pos: S('path', { class: 'lrpc-band' }, gB), neg: S('path', { class: 'lrpc-band lrpc-band--neg' }, gB) });
            out.bands.blocks.push({ x, L, bx, px, nx, y, through: pair(), back: pair(), skip: pair() });
            out.skips.push(tall ? { pos: T(g, (L + px) / 2, y - ARC - 18, '', 'lrpc-skiplabel', 'end'), neg: T(g, (L + px) / 2, y + ARC + 16, '', 'lrpc-skiplabel lrpc-skiplabel--neg', 'start') }
              : { pos: T(g, (L + px) / 2, y - ARC - 16, '', 'lrpc-skiplabel'), neg: T(g, (L + px) / 2, y + ARC + 24, '', 'lrpc-skiplabel lrpc-skiplabel--neg') });
          } else {
            S('circle', { class: 'lrpc-junction', cx: L, cy: y, r: 3.5 }, g);
            const sp = `M${L},${y - 4} L${L},${y - ARC + 10} Q${L},${y - ARC} ${L + 10},${y - ARC} L${px - 10},${y - ARC} Q${px},${y - ARC} ${px},${y - ARC + 10} L${px},${y - 14}`;
            out.skips.push({ path: arrow(g, sp, 'lrpc-skip'), label: tall ? T(g, (L + px) / 2, y - ARC - 8, 'skip', 'lrpc-skiplabel', 'end') : T(g, (L + px) / 2, y - ARC - 7, 'skip', 'lrpc-skiplabel') });
          }
        });
        dir(XS[B] + MS / 2 + 3, XHEAD - 38);
        box(XHEAD, 70, ['head', 'pool, norm,', 'linear']);
        if (rel) out.pixTotal = tall ? T(g, XIN + 4, y - IN / 2 - 8, '', 'lrpc-total', 'end') : T(g, XIN, y + IN / 2 + 18, '', 'lrpc-total');
        return out;
      });
      scene.back = band(S('g', {}, sc), null);
      const cy = (R1 + R2) / 2, gC = S('g', {}, sc);
      scene.cls = Array.from({ length: 10 }, (_, k) => {
        const y = cy + (k - 4.5) * 25, g = S('g', { class: 'lrpc-class' }, gC);
        S('circle', { class: 'lrpw-node', cx: XCLS, cy: y, r: 10 }, g);
        T(g, XCLS, y, String(k), 'lrpw-digit').setAttribute('dominant-baseline', 'central');
        return g;
      });
      arrow(sc, `M${XHEAD + 38},${R1} C${XCLS - 30},${R1} ${XCLS - 40},${cy - 120} ${XCLS - 15},${cy - 115}`, 'lrpc-wire');
      scene.rows.forEach((row) => row.boxes.forEach((bx, k) => {
        Object.entries({ tabindex: 0, role: 'button', 'aria-label': `Open block ${k + 1} below` }).forEach(([a, v]) => bx.g.setAttribute(a, v));
        bx.g.addEventListener('click', () => look(k + 1));
        bx.g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); look(k + 1); } });
      }));
      if (tall) {
        sc.querySelectorAll('.lrpc-box').forEach((b) => { const r = b.querySelector('rect'); upright(b, +r.getAttribute('x') + +r.getAttribute('width') / 2, +r.getAttribute('y') + +r.getAttribute('height') / 2); });
        sc.querySelectorAll('text').forEach((e) => { if (!e.closest('.lrpc-box')) upright(e, +e.getAttribute('x'), +e.getAttribute('y')); });
        sc.querySelectorAll('image').forEach((e) => upright(e, +e.getAttribute('x') + +e.getAttribute('width') / 2, +e.getAttribute('y') + +e.getAttribute('height') / 2));
        Object.assign(pad.style, { left: `${(R1 - IN / 2) / VW * 100}%`, top: `${(XIN - IN / 2 + VTOP) / VH * 100}%`, width: `${IN / VW * 100}%`, height: `${IN / VH * 100}%` });
        Object.assign(clearBtn.style, { left: `${(R1 - IN / 2 - 46) / VW * 100}%`, top: `calc(${(XIN + VTOP) / VH * 100}% - 11px)` });
      } else {
        Object.assign(pad.style, { left: `${(XIN - IN / 2) / W * 100}%`, top: `${(R1 - IN / 2) / HGT * 100}%`, width: `${IN / W * 100}%`, height: `${IN / HGT * 100}%` });
        Object.assign(clearBtn.style, { left: `${XIN / W * 100}%`, top: `calc(${(R1 + IN / 2) / HGT * 100}% + 6px)` });
      }
    };

    const paintForward = () => {
      const row = scene.rows[0], mag = (h) => Float64Array.from({ length: M.N }, (_, p) => { let s = 0; for (let c = 0; c < M.C; c++) s += h[c * M.N + p] ** 2; return Math.sqrt(s); });
      row.input.setAttribute('href', image(f.x, 28, (v) => grey(v)));
      [f.stem, ...f.blocks.map((b) => b.out)].forEach((h, k) => { const m = mag(h), mx = Math.max(...m); row.maps[k].setAttribute('href', image(m, M.G, (v) => grey(v / mx))); });
    };
    const paint = () => {
      const key = `${pred}|${state.cfg.add}|${state.cfg.ln}|${state.cfg.gelu}|${state.cfg.ls}`;
      if (!cache.has(key)) cache.set(key, M.explain(f, pred, state.cfg));
      ex = cache.get(key);
      const row = scene.rows[1], chan = (R) => Float64Array.from({ length: M.N }, (_, p) => { let s = 0; for (let c = 0; c < M.C; c++) s += R[c * M.N + p]; return s; });
      row.input.setAttribute('href', signed(ex.maps[0], 28));
      row.pixTotal.textContent = fmt(sum(ex.maps[0]));
      for (let k = 0; k <= B; k++) {
        const [P, N] = pn(ex.maps[k + 1]);
        row.maps[k].setAttribute('href', signed(chan(ex.maps[k + 1]), M.G)); row.totals[k].textContent = fmt(sum(ex.maps[k + 1]));
        row.parts[k].dataset.pn = `${P},${N}`;
      }
      const bands = row.bands, todo = [[bands.pix, pn(ex.maps[0])], [bands.stem, pn(ex.maps[1])], [bands.top, pn(ex.maps[B + 1])]];
      const big = [...todo.map(([, v]) => v), ...ex.flows.flatMap((t) => [[t.bands.in[0] + t.bands.skip[0], t.bands.in[1] + t.bands.skip[1]], [t.bands.out[0] + t.bands.skip[0], t.bands.out[1] + t.bands.skip[1]]])];
      scale = Math.min(16, 22 / Math.max(1e-9, ...big.map(([P, N]) => Math.max(P, N))));
      todo.forEach(([b, v]) => setBand(b, v));
      ex.flows.forEach((t, k) => {
        const { x, L, bx, px, nx, y, through, back, skip } = bands.blocks[k];
        [[-1, 0, 'pos'], [1, 1, 'neg']].forEach(([s, i, side]) => {
          const wi = width(t.bands.in[i]), wo = width(t.bands.out[i]), ws = width(t.bands.skip[i]), c = (w, under = 0) => y + s * (GAP / 2 + under + w / 2);
          const draw = (el, w, d) => { el.setAttribute('d', d); el.style.strokeWidth = w; el.style.display = w ? '' : 'none'; };
          draw(through[side], wi, `M${nx - MS / 2},${c(wi)} L${bx + 34},${c(wi)}`);
          draw(back[side], wo, `M${bx - 34},${c(wo)} L${x + MS / 2},${c(wo)}`);
          const cr = c(ws, wi), cl = c(ws, wo), top = y + s * ARC, R = clamp(ws / 2 + 4, 6, 14);
          draw(skip[side], ws, `M${nx - MS / 2},${cr} L${px + R},${cr} Q${px},${cr} ${px},${cr + s * R} L${px},${top - s * R} Q${px},${top} ${px - R},${top} L${L + R},${top} Q${L},${top} ${L},${top - s * R} L${L},${cl + s * R} Q${L},${cl} ${L - R},${cl} L${x + MS / 2},${cl}`);
        });
      });
      row.parts.forEach((el) => { const [P, N] = el.dataset.pn.split(',').map(Number); el.textContent = width(N) > 0 ? `+${fmt(P)} and ${fmt(-N)}` : ''; });
      ex.flows.forEach((t, k) => {
        const [, a, b] = row.boxes[k].texts, [P, N] = t.bands.in;
        a.textContent = width(P) > 0 ? `through ${fmt(P)}` : ''; b.textContent = width(N) > 0 ? fmt(-N) : '';
        b.classList.add('lrpc-boxsub--neg');
        const [SP, SN] = t.bands.skip;
        row.skips[k].pos.textContent = width(SP) > 0 ? fmt(SP) : '';
        row.skips[k].neg.textContent = width(SN) > 0 ? fmt(-SN) : '';
      });
      scene.cls.forEach((g, k) => g.classList.toggle('lrpw-explained', k === pred));
      const cy = (R1 + R2) / 2, y = cy + (pred - 4.5) * 25;
      scene.back.shape = (sign, w) => { const o = sign * (GAP / 2 + w / 2); return `M${XCLS},${y} C${XCLS - 40},${y} ${XCLS - 30},${R2 + o} ${XHEAD + 35},${R2 + o}`; };
      setBand(scene.back, [Math.max(0, ex.start), Math.max(0, -ex.start)]);
      if (!state.look) state.look = 1 + ex.flows.reduce((best, t, k) => (Math.abs(t.in) > Math.abs(ex.flows[best].in) ? k : best), 0);
      paintCards();
      readout();
    };

    const readout = () => {
      const c = state.cfg, parts = [];
      const tops = ex.maps.slice(1).map(sum), pix = sum(ex.maps[0]);
      parts.push(`The network predicts <b>${pred}</b>, and the relevance starts there with <b>${fmt(ex.start)}</b>.`);
      const none = Object.keys(RULES).filter((k) => c[k] !== RULES[k]);
      const shares = ex.flows.filter((t) => Math.abs(t.f2) > 1e-9).map((t) => Math.round(100 * (1 - t.gelu / t.f2)));
      const lost = [Math.min(...shares), Math.max(...shares)];
      parts.push(c.add === 'split' ? '<b>The residual connections</b> split the relevance. Each is an addition, a neuron whose two weights are 1, so its rule gives the skip and the block the share each contributed, and nothing is created.'
        : `<b>The residual connections</b> without their rule follow the gradient. The gradient of a sum is 1 for both inputs, so the skip and the block each receive a full copy, and the total doubles at every block, up to <b>${fmt(tops[0])}</b> after the stem.`);
      parts.push(c.ln === 'rule' ? '<b>LayerNorm</b> holds its spread fixed, so the relevance passes back through it like a weighted layer and keeps its total.'
        : '<b>LayerNorm</b> without its rule follows its gradient, which turns what reaches it into positive and negative amounts that add up to zero. The norm in the head does this to the whole start, and every block does it to its own share. The totals that remain below are leftovers of amounts that no longer cancel exactly, and the maps are noise.');
      parts.push(c.gelu === 'rule' ? '<b>GELU</b> passes the relevance on unchanged, because it changes each value on its own and mixes nothing.'
        : `<b>GELU</b> without its rule follows the gradient and multiplies the relevance by its slope. The slope is close to zero for negative inputs${c.ln === 'rule' ? `, so each block loses ${lost[0]} to ${lost[1]} percent of what enters it` : ''}.`);
      parts.push(c.ls === 'rule' ? '<b>Layer scale</b> passes the relevance on unchanged, because it only rescales what the block computed.'
        : '<b>Layer scale</b> without its rule follows the gradient and multiplies the relevance by its learned numbers γ. Channels with a small γ shrink, those with a large one grow, and negative ones flip their sign, so the amount drifts from block to block.');
      if (!none.length) parts.push(`The blocks keep the total, and <b>${fmt(tops[0])}</b> reaches the stem. The stem loses more. Where no pixel contributes positively to a cell, the z⁺ rule has nothing to follow and drops that cell's relevance, so <b>${fmt(pix)}</b> arrives at the pixels.`);
      else if (c.ln === 'rule') {
        const key = `${pred}|split|rule|rule|rule`;
        if (!cache.has(key)) cache.set(key, M.explain(f, pred, RULES));
        parts.push(`The maps keep their shape, but <b>${fmt(pix)}</b> arrives at the pixels, against <b>${fmt(sum(cache.get(key).maps[0]))}</b> with every rule.`);
      }
      parts.push('Click or tap any block to see what happens to the relevance inside it.');
      readoutEl.innerHTML = parts.join(' ');
    };

    const CW = 300, CH = 206, TOP = 30, BOT = 150;
    const OPS = [
      { op: 'add', name: 'Residual', vals: ['split', 'copy'], labels: ['Split', 'Copy'], reach: 'reaches it', note: 'A residual connection is an addition, a neuron whose two weights are 1. The rule splits the relevance, giving the skip and the block the share each contributed. Without it, the relevance follows the gradient, which is 1 for both, so each gets a full copy.' },
      { op: 'ln', name: 'LayerNorm', vals: ['rule', 'none'], labels: ['Fixed spread', 'Gradient'], reach: 'reaches it', note: 'Each bar is one channel, added up over all cells. Without the rule, the bars always add up to zero.' },
      { op: 'gelu', name: 'GELU', vals: ['rule', 'none'], labels: ['Pass on', '× slope'], reach: 'reaches it', note: 'Each bar adds up the units whose value entering GELU falls in its range. Without the rule, their relevance is multiplied by the slope of GELU, which is close to zero for negative values.' },
      { op: 'ls', name: 'Layer scale', vals: ['rule', 'none'], labels: ['Pass on', '× γ'], reach: 'reaches it', note: 'Each bar is one channel. Without the rule, its relevance is multiplied by the number the channel learned, and some of these numbers are negative.' },
    ];
    const cards = {};
    const buildCards = () => {
      cardsEl.replaceChildren();
      OPS.forEach((o) => {
        const el = document.createElement('div'), head = document.createElement('p'), note = document.createElement('p');
        el.className = 'lrpc-card'; head.className = 'lrpc-card-head'; note.className = 'lrpc-card-note'; note.textContent = o.note;
        const plot = S('svg', { viewBox: `0 0 ${CW} ${CH}`, class: 'lrpc-card-plot', role: 'img', 'aria-label': `${o.name} with and without its rule` });
        el.append(head, plot, note); cardsEl.append(el);
        cards[o.op] = { head, plot };
      });
    };
    const pane = (svg, o, i) => {
      const g = S('g', { class: state.cfg[o.op] === o.vals[i] ? 'lrpc-pane lrpc-pane--on' : 'lrpc-pane' }, svg);
      g.addEventListener('click', () => setOp(o.op, o.vals[i]));
      return g;
    };
    const axis = (vals) => {
      const mp = Math.max(0, ...vals), mn = Math.max(0, ...vals.map((v) => -v)), k = (BOT - TOP) / Math.max(1e-12, mp + mn);
      const base = TOP + k * mp;
      return { base, y: (v) => base - k * v };
    };
    const bars = (g, vals, x0, w, ax, ticks) => {
      const bw = w / vals.length;
      S('line', { x1: x0, x2: x0 + w, y1: ax.base, y2: ax.base, class: 'lrpc-axis' }, g);
      vals.forEach((v, i) => {
        const y = ax.y(v), h = Math.abs(y - ax.base);
        if (h >= 0.4) S('rect', { x: x0 + i * bw + bw * 0.18, y: Math.min(y, ax.base), width: bw * 0.64, height: h, rx: Math.min(2, bw * 0.15), class: v >= 0 ? 'lrpc-bar' : 'lrpc-bar lrpc-bar--neg' }, g);
        if (ticks) T(g, x0 + (i + 0.5) * bw, BOT + 14, ticks[i], 'lrpc-card-tick');
      });
    };
    const paintBars = (plot, o, d) => {
      const half = CW / 2, sets = o.op === 'add' ? [d.split, d.copy] : [d.rule, d.none].map((a) => Array.from(a));
      const ax = axis([...sets.flat(), ...(o.op === 'add' ? [d.in] : [])]);
      sets.forEach((vals, i) => {
        const g = pane(plot, o, i), x0 = i * half;
        S('rect', { x: x0, y: 0, width: half, height: CH, class: 'lrpc-hit' }, g);
        T(g, x0 + half / 2, 14, o.labels[i], 'lrpc-card-label');
        bars(g, vals, x0 + 16, half - 32, ax, o.op === 'add' ? ['skip', 'block'] : null);
        if (o.op === 'add') S('line', { x1: x0 + 10, x2: x0 + half - 10, y1: ax.y(d.in), y2: ax.y(d.in), class: 'lrpc-reach' }, g);
        T(g, x0 + half / 2, CH - 6, `adds up to ${fmt(sum(vals))}`, 'lrpc-card-total');
      });
      S('line', { x1: half, x2: half, y1: 4, y2: CH - 24, class: 'lrpc-divider' }, plot);
    };
    const paintGelu = (plot, o, d) => {
      const half = CW / 2, ax = axis([...d.rule, ...d.none]);
      [d.rule, d.none].forEach((vals, i) => {
        const g = pane(plot, o, i), x0 = i * half + 16, w = half - 32, X = (z) => x0 + w * (z - ZLO) / (ZHI - ZLO);
        S('rect', { x: i * half, y: 0, width: half, height: CH, class: 'lrpc-hit' }, g);
        T(g, i * half + half / 2, 14, o.labels[i], 'lrpc-card-label');
        if (i) {
          let path = '';
          for (let s2 = 0; s2 <= 60; s2++) { const z = ZLO + (ZHI - ZLO) * s2 / 60; path += `${s2 ? 'L' : 'M'}${X(z).toFixed(1)},${(ax.base - (ax.base - TOP) * clamp(slope(z), 0, 1.2) / 1.2).toFixed(1)}`; }
          S('path', { d: path, class: 'lrpc-curve' }, g);
          T(g, X(ZHI), TOP - 2, 'slope', 'lrpc-card-tick', 'end');
        }
        bars(g, Array.from(vals), x0, w, ax, null);
        [-2, 0, 2].forEach((z) => T(g, X(z), BOT + 14, String(z).replace('-', '−'), 'lrpc-card-tick'));
        T(g, i * half + half / 2, CH - 6, `adds up to ${fmt(sum(vals))}`, 'lrpc-card-total');
      });
      S('line', { x1: half, x2: half, y1: 4, y2: CH - 24, class: 'lrpc-divider' }, plot);
      T(plot, half, BOT + 30, 'value entering GELU', 'lrpc-card-tick');
    };
    const paintCards = () => {
      const k = state.look, t = ex.flows[k - 1];
      scene.rows.forEach((row) => row.boxes.forEach((bx, i) => bx.g.classList.toggle('lrpc-open', i === k - 1)));
      const arrive = ex.maps[k + 1], after = ex.maps[k], both = ([P, N]) => (width(N) > 0 && width(P) > 0 ? ` (+${fmt(P)} and ${fmt(-N)})` : '');
      panelText.innerHTML = [`<b class="lrpc-pick">Selected block ${k}:</b> <b>${fmt(sum(arrive))}</b>${both(pn(arrive))} arrives at its plus sign. The skip takes <b>${fmt(t.skip)}</b>${both(t.bands.skip)}, and <b>${fmt(t.in)}</b>${both(t.bands.in)} enters the block.`,
        `Going back through it, <b>layer scale</b> passes on <b>${fmt(t.ls)}</b>, the <b>second linear layer</b> <b>${fmt(t.f2)}</b>, <b>GELU</b> <b>${fmt(t.gelu)}</b>, the <b>first linear layer</b> <b>${fmt(t.f1)}</b>, and <b>LayerNorm</b> <b>${fmt(t.ln)}</b>.`,
        `The <b>convolution</b> hands <b>${fmt(t.dw)}</b> back to the stream, and together with the skip, <b>${fmt(sum(after))}</b> arrives ${k > 1 ? `after block ${k - 1}` : 'after the stem'}.`].join(' ');
      summaryEl.textContent = `What each operation in block ${k} does with the relevance`;
      OPS.forEach((o) => {
        const { head, plot } = cards[o.op], d = t.card[o.op];
        head.innerHTML = `<b>${o.name}</b> in block ${k}. <b>${fmt(d.in)}</b> ${o.reach}.`;
        plot.replaceChildren();
        (o.op === 'gelu' ? paintGelu : paintBars)(plot, o, d);
      });
    };
    const look = (k) => { if (state.look === k) return; state.look = k; paintCards(); };

    const load = () => {
      f = M.forward(state.pixels); cache.clear(); state.look = null;
      pred = f.logits.indexOf(Math.max(...f.logits));
      paintForward(); paint();
    };
    const setOp = (op, val) => {
      state.cfg[op] = val;
      root.querySelectorAll(`.lrpw-op[data-op=${op}]`).forEach((o) => o.setAttribute('aria-pressed', String(o.dataset.val === val)));
      paint();
    };
    const steady = (el, fn) => { const y = el.getBoundingClientRect().top; fn(); window.scrollBy(0, el.getBoundingClientRect().top - y); };
    root.querySelectorAll('.lrpw-op').forEach((b) => b.addEventListener('click', () => steady(b, () => setOp(b.dataset.op, b.dataset.val))));

    const pctx = pad.getContext('2d'), P = pad.width;
    const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const ink = canvas(P, P), ictx = ink.getContext('2d');
    const blank = () => { ictx.fillStyle = '#000'; ictx.fillRect(0, 0, P, P); };
    blank();
    { const c = canvas(20, 20), x = c.getContext('2d'), im = x.createImageData(20, 20);
      for (let q = 0; q < 400; q++) { const v = state.pixels[(4 + ((q / 20) | 0)) * 28 + 4 + (q % 20)]; im.data.set([v, v, v, 255], q * 4); }
      x.putImageData(im, 0, 0); ictx.imageSmoothingEnabled = true; ictx.drawImage(c, 0, 0, P, P); }
    const toPixels = () => {
      let c = ink;
      while (c.width > 56) { const n = canvas(Math.ceil(c.width / 2), Math.ceil(c.height / 2)), nx = n.getContext('2d'); nx.imageSmoothingQuality = 'high'; nx.drawImage(c, 0, 0, n.width, n.height); c = n; }
      const o = canvas(28, 28), ox = o.getContext('2d');
      ox.fillStyle = '#000'; ox.fillRect(0, 0, 28, 28); ox.imageSmoothingQuality = 'high'; ox.drawImage(c, 4, 4, 20, 20);
      const dd = ox.getImageData(0, 0, 28, 28).data;
      return Array.from({ length: 784 }, (_, i) => dd[i * 4]);
    };
    let drawing = false, last = null;
    const at = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * P, (e.clientY - r.top) / r.height * P]; };
    const stroke = (a, b) => { for (const [ctx, colour] of [[ictx, '#fff'], [pctx, '#26302b']]) { ctx.strokeStyle = colour; ctx.lineWidth = 30; ctx.lineCap = ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); } };
    pad.addEventListener('pointerdown', (e) => { drawing = true; pad.setPointerCapture(e.pointerId); last = at(e); stroke(last, last); });
    pad.addEventListener('pointermove', (e) => { if (!drawing) return; const p = at(e); stroke(last, p); last = p; });
    const finish = () => { if (!drawing) return; drawing = false; state.pixels = toPixels(); steady(pad, load); pctx.clearRect(0, 0, P, P); };
    pad.addEventListener('pointerup', finish);
    pad.addEventListener('pointercancel', finish);
    clearBtn.addEventListener('click', () => steady(clearBtn, () => { blank(); state.pixels = new Array(784).fill(0); load(); readoutEl.textContent = 'Draw a digit on the input, filling the square.'; }));

    build();
    buildCards();
    load();
    narrow.addEventListener('change', () => { build(); load(); });
  };

  const init = () => document.querySelectorAll('[data-lrp-cnx]').forEach((el) => { if (el.dataset.ready) return; el.dataset.ready = 'true'; figure(el); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

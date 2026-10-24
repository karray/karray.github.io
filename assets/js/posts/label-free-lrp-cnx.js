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
  const RULES = { add: 'split', ln: 'rule', gelu: 'rule', ls: 'rule' };

  const model = (d) => {
    const C = d.channels, G = d.grid, N = G * G, H = 4 * C;
    const norm = (x, gam) => {
      let m = 0; for (const v of x) m += v; m /= x.length;
      let s = 0; for (const v of x) s += (v - m) ** 2;
      const sd = Math.sqrt(s / x.length + 1e-5);
      return { m, sd, n: x.map((v, i) => gam[i] * (v - m) / sd) };
    };
    const zplus = (a, W, R, I, O, F) => {
      const out = new Float64Array(I);
      for (let o = 0; o < O; o++) {
        if (!R[o]) continue;
        let z = 0; for (let i = 0; i < I; i++) { const c = a[i] * W[o * I + i]; if (c > 0) z += c; }
        if (z <= 0) continue;
        for (let i = 0; i < I; i++) { const c = a[i] * W[o * I + i]; if (c > 0) { const v = c * R[o] / z; out[i] += v; if (F) F[o][i] += v; } }
      }
      return out;
    };
    const normBack = (x, gam, Rn, mode, F) => {
      const n = x.length, { m, sd } = norm(x, gam), out = new Float64Array(n);
      const add = (i, j, v) => { out[j] += v; if (F) F[i][j] += v; };
      for (let i = 0; i < n; i++) {
        if (!Rn[i]) continue;
        if (mode === 'none') { const xi = (x[i] - m) / sd; for (let j = 0; j < n; j++) add(i, j, gam[i] / sd * ((i === j) - 1 / n - xi * (x[j] - m) / sd / n) * Rn[i]); continue; }
        const ni = gam[i] * (x[i] - m) / sd;
        if (Math.abs(ni) < 1e-12) continue;
        for (let j = 0; j < n; j++) add(i, j, gam[i] / sd * ((i === j) - 1 / n) * x[j] * Rn[i] / ni);
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
        const vec = (n) => new Float64Array(n), mat = (a, b2) => [...Array(a)].map(() => new Float64Array(b2));
        const agg = { Rin: vec(C), Rx: vec(C), Rv: vec(C), Ru: vec(C), F2: mat(C, H), Rg: vec(H), Rz: vec(H), F1: mat(H, C), Rn: vec(C), FL: mat(C, C), Rd: vec(C), Rw: vec(C) };
        for (let p = 0; p < N; p++) {
          const Ru = Float64Array.from({ length: C }, (_, c) => (cfg.ls === 'rule' ? 1 : b.gamma[c]) * Rv[c * N + p]);
          const Rg = zplus(st.g[p], b.f2, Ru, H, C, agg.F2);
          const Rz = cfg.gelu === 'rule' ? Rg : Rg.map((r, o) => slope(st.z[p][o]) * r);
          const Rn = zplus(st.n[p], b.f1, Rz, C, H, agg.F1);
          const Rdp = normBack(col(st.d, p), b.ln, Rn, cfg.ln, agg.FL);
          t.ls += sum(Ru); t.f2 += sum(Rg); t.gelu += sum(Rz); t.f1 += sum(Rn); t.ln += sum(Rdp);
          for (let c = 0; c < C; c++) { Rd[c * N + p] = Rdp[c]; agg.Rin[c] += Rh[c * N + p]; agg.Rx[c] += Rx[c * N + p]; agg.Rv[c] += Rv[c * N + p]; agg.Ru[c] += Ru[c]; agg.Rn[c] += Rn[c]; agg.Rd[c] += Rdp[c]; }
          for (let o = 0; o < H; o++) { agg.Rg[o] += Rg[o]; agg.Rz[o] += Rz[o]; }
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
        for (let c = 0; c < C; c++) for (let p = 0; p < N; p++) agg.Rw[c] += Rw[c * N + p];
        t.agg = agg;
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

  const W = 1400, HGT = 486, R1 = 132, R2 = 404, MS = 76, IN = 88, GAP = 6, ARC = 52;
  const XS = [230, 470, 710, 950, 1190], XIN = 52, XSTEM = 146, XHEAD = 1282, XCLS = 1368;

  const figure = async (root) => {
    const svg = root.querySelector('.lrpc-scene'), pad = root.querySelector('.lrpw-pad'), clearBtn = root.querySelector('.lrpw-clear'), readoutEl = root.querySelector('.lrpw-readout');
    let data;
    try { data = await (await fetch(root.dataset.src)).json(); } catch (e) { readoutEl.textContent = 'The network could not be loaded.'; return; }
    const M = model(data), B = data.blocks.length;
    const state = { pixels: data.start, cfg: { ...RULES }, classHover: null, look: null };
    const panel = root.querySelector('.lrpc-inside'), panelText = root.querySelector('.lrpc-inside-text');
    let f, pred, cls, ex;
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
      const defs = S('defs', {}, svg), mk = S('marker', { id: 'lrpc-head', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M0,0 L10,5 L0,10 z', class: 'lrpc-arrowhead' }, mk);
      T(svg, 8, 30, 'Prediction', 'lrpc-row', 'start');
      T(svg, 8, R2 - 98, 'Explanation', 'lrpc-row lrpc-row--rel', 'start');
      scene.rows = [R1, R2].map((y, r) => {
        const g = S('g', {}, svg), rel = r === 1, out = { maps: [], boxes: [], totals: [], parts: [], skips: [], bands: { blocks: [] } };
        const gB = S('g', {}, g), lane = (sign, w) => y + sign * (GAP / 2 + w / 2), hz = (a, b) => (sign, w) => `M${a},${lane(sign, w)} L${b},${lane(sign, w)}`;
        const sq = (x, size, label) => {
          const img = S('image', { x: x - size / 2, y: y - size / 2, width: size, height: size, preserveAspectRatio: 'none', style: 'image-rendering: pixelated' }, g);
          S('rect', { class: 'lrpc-frame', x: x - size / 2, y: y - size / 2, width: size, height: size }, g);
          if (label) T(g, x, y - size / 2 - 8, label, 'lrpc-label');
          return img;
        };
        const box = (x, w, lines, cls = '') => {
          const bg = S('g', { class: `lrpc-box ${cls}` }, g);
          S('rect', { x: x - w / 2, y: y - 24, width: w, height: 48, rx: 8 }, bg);
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
          out.totals.push(rel ? T(g, x - MS / 2, y + MS / 2 + 18, '', 'lrpc-total', 'start') : null);
          out.parts.push(rel ? T(g, x - MS / 2, y + MS / 2 + 32, '', 'lrpc-parts', 'start') : null);
          if (k === B) return;
          const L = x + MS / 2 + 20, bx = x + 114, px = x + 172, nx = XS[k + 1];
          dir(x + MS / 2 + 3, bx - 39);
          out.boxes.push(box(bx, 70, rel ? [`block ${k + 1}`, '', ''] : [`block ${k + 1}`], 'lrpc-block'));
          dir(bx + 39, px - 14);
          if (!rel) { S('circle', { class: 'lrpc-plus', cx: px, cy: y, r: 11 }, g); T(g, px, y + 5, '+', 'lrpc-plussign'); }
          if (rel) S('circle', { cx: px, cy: y, r: 15, fill: 'transparent', style: 'cursor: pointer' }, g).addEventListener('click', () => look(k + 1));
          dir(px + 14, nx - MS / 2 - 3);
          if (rel) {
            const pair = () => ({ pos: S('path', { class: 'lrpc-band' }, gB), neg: S('path', { class: 'lrpc-band lrpc-band--neg' }, gB) });
            out.bands.blocks.push({ x, L, bx, px, nx, y, through: pair(), back: pair(), skip: pair() });
            out.skips.push({ pos: T(g, (L + px) / 2, y - ARC - 16, '', 'lrpc-skiplabel'), neg: T(g, (L + px) / 2, y + ARC + 24, '', 'lrpc-skiplabel lrpc-skiplabel--neg') });
          } else {
            S('circle', { class: 'lrpc-junction', cx: L, cy: y, r: 3.5 }, g);
            const sp = `M${L},${y - 4} L${L},${y - ARC + 10} Q${L},${y - ARC} ${L + 10},${y - ARC} L${px - 10},${y - ARC} Q${px},${y - ARC} ${px},${y - ARC + 10} L${px},${y - 14}`;
            out.skips.push({ path: arrow(g, sp, 'lrpc-skip'), label: T(g, (L + px) / 2, y - ARC - 7, k ? '' : 'skip', 'lrpc-skiplabel') });
          }
        });
        dir(XS[B] + MS / 2 + 3, XHEAD - 38);
        box(XHEAD, 70, ['head', 'pool, norm,', 'linear']);
        if (rel) out.pixTotal = T(g, XIN, y + IN / 2 + 18, '', 'lrpc-total');
        return out;
      });
      scene.back = band(S('g', {}, svg), null);
      const cy = (R1 + R2) / 2, gC = S('g', {}, svg);
      scene.cls = Array.from({ length: 10 }, (_, k) => {
        const y = cy + (k - 4.5) * 25, g = S('g', { class: 'lrpc-class' }, gC);
        S('circle', { class: 'lrpw-node', cx: XCLS, cy: y, r: 10 }, g);
        T(g, XCLS, y + 4, String(k), 'lrpw-digit');
        const hit = S('rect', { x: XCLS - 22, y: y - 12.5, width: 44, height: 25, fill: 'transparent', style: 'cursor: pointer' }, gC);
        hit.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') showClass(k); });
        hit.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') showClass(null); });
        hit.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') showClass(state.classHover === k ? null : k); });
        return g;
      });
      arrow(svg, `M${XHEAD + 38},${R1} C${XCLS - 30},${R1} ${XCLS - 40},${cy - 120} ${XCLS - 15},${cy - 115}`, 'lrpc-wire');
      scene.rows.forEach((row) => row.boxes.forEach((bx, k) => {
        Object.entries({ tabindex: 0, role: 'button', 'aria-label': `Open block ${k + 1} below` }).forEach(([a, v]) => bx.g.setAttribute(a, v));
        bx.g.addEventListener('click', () => look(k + 1));
        bx.g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); look(k + 1); } });
      }));
    };

    const paintForward = () => {
      const row = scene.rows[0], mag = (h) => Float64Array.from({ length: M.N }, (_, p) => { let s = 0; for (let c = 0; c < M.C; c++) s += h[c * M.N + p] ** 2; return Math.sqrt(s); });
      row.input.setAttribute('href', image(f.x, 28, (v) => grey(v)));
      [f.stem, ...f.blocks.map((b) => b.out)].forEach((h, k) => { const m = mag(h), mx = Math.max(...m); row.maps[k].setAttribute('href', image(m, M.G, (v) => grey(v / mx))); });
    };
    const paint = () => {
      const key = `${cls}|${state.cfg.add}|${state.cfg.ln}|${state.cfg.gelu}|${state.cfg.ls}`;
      if (!cache.has(key)) cache.set(key, M.explain(f, cls, state.cfg));
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
      row.parts.forEach((el) => { const [P, N] = el.dataset.pn.split(',').map(Number); el.textContent = width(N) >= 2 ? `+${fmt(P)} and ${fmt(-N)}` : ''; });
      ex.flows.forEach((t, k) => {
        const [, a, b] = row.boxes[k].texts, [P, N] = t.bands.in;
        a.textContent = width(P) >= 2 ? `through ${fmt(P)}` : ''; b.textContent = width(N) >= 2 ? fmt(-N) : '';
        b.classList.add('lrpc-boxsub--neg');
        const [SP, SN] = t.bands.skip;
        row.skips[k].pos.textContent = width(SP) >= 2 ? fmt(SP) : '';
        row.skips[k].neg.textContent = width(SN) >= 2 ? fmt(-SN) : '';
      });
      scene.cls.forEach((g, k) => { g.classList.toggle('lrpw-explained', k === cls); g.classList.toggle('lrpw-hover', k === state.classHover); });
      const cy = (R1 + R2) / 2, y = cy + (cls - 4.5) * 25;
      scene.back.shape = (sign, w) => { const o = sign * (GAP / 2 + w / 2); return `M${XCLS},${y} C${XCLS - 40},${y} ${XCLS - 30},${R2 + o} ${XHEAD + 35},${R2 + o}`; };
      setBand(scene.back, [Math.max(0, ex.start), Math.max(0, -ex.start)]);
      if (!state.look) state.look = 1 + ex.flows.reduce((best, t, k) => (Math.abs(t.in) > Math.abs(ex.flows[best].in) ? k : best), 0);
      paintInside();
      readout();
    };

    const readout = () => {
      const c = state.cfg, parts = [];
      const tops = ex.maps.slice(1).map(sum), pix = sum(ex.maps[0]);
      if (state.classHover !== null) parts.push(`<b>Class ${cls}</b> has a score of <b>${f.logits[cls].toFixed(1).replace('-', '−')}</b>${cls !== pred ? `, and the network predicts ${pred}` : ''}.`);
      parts.push(`The relevance starts with <b>${fmt(ex.start)}</b> at class ${cls}.`);
      const none = Object.keys(RULES).filter((k) => c[k] !== RULES[k]);
      parts.push(c.add === 'split' ? '<b>The additions</b> split the relevance between the skip and the block, in proportion to what each added, so nothing is created.'
        : `<b>The additions</b> copy, so the skip and the block each receive the full amount, and the total doubles at every block, up to <b>${fmt(tops[0])}</b> after the stem.`);
      parts.push(c.ln === 'rule' ? '<b>LayerNorm</b> holds its spread fixed, so the relevance passes back through it like a layer with weights and keeps its total.'
        : '<b>LayerNorm</b> without its rule turns what reaches it into positive and negative amounts that add up to zero. The norm in the head does this to the whole start, and what remains after it is noise.');
      parts.push(c.gelu === 'rule' ? '<b>GELU</b> passes the relevance on unchanged, because it changes each value on its own and mixes nothing.'
        : '<b>GELU</b> without its rule multiplies the relevance by its slope, and a little leaks away in every block.');
      parts.push(c.ls === 'rule' ? '<b>Layer scale</b> passes the relevance on unchanged, because it only rescales what the block computed.'
        : '<b>Layer scale</b> without its rule multiplies the relevance by its learned numbers, some of them negative, and part of what enters each block cancels.');
      if (!none.length) parts.push(`The stream holds <b>${fmt(tops[B])}</b> after every block. Only the first block and the last step lose a little, where the z⁺ rule finds no positive contribution to follow, and <b>${fmt(pix)}</b> arrives at the pixels.`);
      else parts.push(c.ln === 'rule' ? `The maps still look plausible, but <b>${fmt(pix)}</b> arrives at the pixels.` : `<b>${fmt(pix)}</b> arrives at the pixels.`);
      parts.push('Click a block to open it below.');
      readoutEl.innerHTML = parts.join(' ');
    };

    const IX = { skip: 120, w: 300, d: 450, n: 600, z: 780, g: 900, u: 1070, v: 1190, in: 1320 }, PY0 = 78, PY1 = 282, PH = 344;
    const pane = {};
    const buildInside = () => {
      panel.replaceChildren();
      pane.title = T(panel, 0, 22, '', 'lrpc-row lrpc-row--small', 'start');
      T(panel, 0, 42, 'Click another block above to open it. Each line carries the relevance along one connection, added up over all cells of the grid.', 'lrpc-label', 'start');
      const gE = S('g', {}, panel), gN = S('g', {}, panel), ys = (n) => Array.from({ length: n }, (_, i) => PY0 + (i + 0.5) * (PY1 - PY0) / n);
      const y16 = ys(M.C), y64 = ys(M.H);
      const col = (key, n, label, sub) => {
        const yy = n === M.C ? y16 : y64, h = n === M.C ? 9 : 2.4;
        T(panel, IX[key], PH - 12, label, 'lrpc-label');
        if (sub) T(panel, IX[key], PH, sub, 'lrpc-label');
        pane[`t_${key}`] = T(panel, IX[key], PY1 + 22, '', 'lrpc-total');
        if (n !== M.C) S('rect', { class: 'lrpc-frame', x: IX[key] - 7, y: PY0 - 1, width: 14, height: PY1 - PY0 + 2 }, gN);
        return yy.map((y) => S('rect', { class: n === M.C ? 'lrpc-unit' : 'lrpc-unit lrpc-unit--small', x: IX[key] - 6, y: y - h / 2, width: 12, height: h }, gN));
      };
      pane.nodes = { in: col('in', M.C, 'at the plus sign'), v: col('v', M.C, 'into the block'), u: col('u', M.C, 'layer scale'), g: col('g', M.H, 'second linear'), z: col('z', M.H, 'GELU'),
        n: col('n', M.C, 'first linear'), d: col('d', M.C, 'LayerNorm'), w: col('w', M.C, 'convolution', 'back to the stream'), skip: col('skip', M.C, 'skip', 'past the block') };
      const line = (x1, y1, x2, y2) => S('path', { class: 'lrpc-edge', d: `M${x1 + 6},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2 - 6},${y2}` }, gE);
      const back = (a, b, ya, yb, na, nb) => Array.from({ length: na }, (_, i) => Array.from({ length: nb }, (_, j) => line(IX[b], yb[j], IX[a], ya[i])));
      pane.edges = {
        v: y16.map((y) => line(IX.v, y, IX.in, y)), u: y16.map((y) => line(IX.u, y, IX.v, y)),
        g: back('u', 'g', y16, y64, M.C, M.H), z: y64.map((y) => line(IX.z, y, IX.g, y)),
        n: back('z', 'n', y64, y16, M.H, M.C), d: back('n', 'd', y16, y16, M.C, M.C), w: y16.map((y) => line(IX.w, y, IX.d, y)),
        skip: y16.map((y, j) => S('path', { class: 'lrpc-edge', d: `M${IX.in},${y - 5} L${IX.in},${58 + j * 1.2} L${IX.skip},${58 + j * 1.2} L${IX.skip},${y - 5}` }, gE)),
      };
    };
    const paintInside = () => {
      const k = state.look, r = ex.flows[k - 1].agg;
      pane.title.textContent = `Inside block ${k}: the relevance that reaches it, followed back through each of its layers`;
      scene.rows.forEach((row) => row.boxes.forEach((bx, i) => bx.g.classList.toggle('lrpc-open', i === k - 1)));
      const top = (...arrs) => Math.max(1e-15, ...arrs.flatMap((a) => Array.from(a, Math.abs)));
      const setE = (el, v, mx) => { const w = 4.5 * Math.abs(v) / mx; el.style.strokeWidth = w; el.style.display = w >= 0.15 ? '' : 'none'; el.classList.toggle('lrpc-edge--neg', v < 0); };
      const split = top(r.Rv, r.Rx), m2 = top(...r.F2), m1 = top(...r.F1), mL = top(...r.FL);
      r.Rv.forEach((v, i) => setE(pane.edges.v[i], v, split)); r.Rx.forEach((v, i) => setE(pane.edges.skip[i], v, split));
      r.Ru.forEach((v, i) => setE(pane.edges.u[i], v, top(r.Ru)));
      r.F2.forEach((row, i) => row.forEach((v, j) => setE(pane.edges.g[i][j], v, m2))); r.Rz.forEach((v, i) => setE(pane.edges.z[i], v, top(r.Rz)));
      r.F1.forEach((row, i) => row.forEach((v, j) => setE(pane.edges.n[i][j], v, m1))); r.FL.forEach((row, i) => row.forEach((v, j) => setE(pane.edges.d[i][j], v, mL)));
      r.Rw.forEach((v, i) => setE(pane.edges.w[i], v, top(r.Rw)));
      const fill = (nodes, vals) => { let m = 1e-12; for (const v of vals) m = Math.max(m, Math.abs(v)); nodes.forEach((el, i) => { const t = vals[i] / m; el.style.fill = `rgb(${(t >= 0 ? red(t) : blue(-t)).join(',')})`; }); };
      const t = ex.flows[k - 1], arrive = ex.maps[k + 1], after = ex.maps[k], both = ([P, N]) => (width(N) >= 2 && width(P) >= 2 ? ` (+${fmt(P)} and ${fmt(-N)})` : '');
      panelText.innerHTML = [`<b>${fmt(sum(arrive))}</b>${both(pn(arrive))} arrives at the plus sign after <b>block ${k}</b>, opened below. The skip takes <b>${fmt(t.skip)}</b>${both(t.bands.skip)}, and <b>${fmt(t.in)}</b>${both(t.bands.in)} enters the block.`,
        `Going back through it, <b>layer scale</b> passes on <b>${fmt(t.ls)}</b>, the <b>second linear layer</b> <b>${fmt(t.f2)}</b>, <b>GELU</b> <b>${fmt(t.gelu)}</b>, the <b>first linear layer</b> <b>${fmt(t.f1)}</b>, and <b>LayerNorm</b> <b>${fmt(t.ln)}</b>.`,
        `The <b>convolution</b> hands <b>${fmt(t.dw)}</b> back to the stream, and together with the skip, <b>${fmt(sum(after))}</b> arrives ${k > 1 ? `after block ${k - 1}` : 'after the stem'}.`].join(' ');
      [['in', r.Rin], ['v', r.Rv], ['u', r.Ru], ['g', r.Rg], ['z', r.Rz], ['n', r.Rn], ['d', r.Rd], ['w', r.Rw], ['skip', r.Rx]].forEach(([key, vals]) => { fill(pane.nodes[key], vals); pane[`t_${key}`].textContent = fmt(sum(vals)); });
    };
    const look = (k) => { if (state.look === k) return; state.look = k; paintInside(); };

    const showClass = (k) => { state.classHover = k; cls = k ?? pred; paint(); };
    const load = () => {
      f = M.forward(state.pixels); cache.clear(); state.look = null;
      pred = cls = f.logits.indexOf(Math.max(...f.logits)); state.classHover = null;
      paintForward(); paint();
    };
    root.querySelectorAll('.lrpw-op').forEach((b) => b.addEventListener('click', () => {
      state.cfg[b.dataset.op] = b.dataset.val;
      root.querySelectorAll(`.lrpw-op[data-op=${b.dataset.op}]`).forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
      paint();
    }));

    const pctx = pad.getContext('2d'), P = pad.width;
    const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const ink = canvas(P, P), ictx = ink.getContext('2d');
    const blank = () => { ictx.fillStyle = '#000'; ictx.fillRect(0, 0, P, P); };
    Object.assign(pad.style, { left: `${(XIN - IN / 2) / W * 100}%`, top: `${(R1 - IN / 2) / HGT * 100}%`, width: `${IN / W * 100}%`, height: `${IN / HGT * 100}%` });
    Object.assign(clearBtn.style, { left: `${XIN / W * 100}%`, top: `calc(${(R1 + IN / 2) / HGT * 100}% + 6px)` });
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
    const finish = () => { if (!drawing) return; drawing = false; state.pixels = toPixels(); load(); pctx.clearRect(0, 0, P, P); };
    pad.addEventListener('pointerup', finish);
    pad.addEventListener('pointercancel', finish);
    clearBtn.addEventListener('click', () => { blank(); state.pixels = new Array(784).fill(0); load(); readoutEl.textContent = 'Draw a digit on the input, filling the square.'; });

    build();
    buildInside();
    load();
  };

  const init = () => document.querySelectorAll('[data-lrp-cnx]').forEach((el) => { if (el.dataset.ready) return; el.dataset.ready = 'true'; figure(el); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

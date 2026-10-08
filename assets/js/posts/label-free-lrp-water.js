/* The water figure of the LRP post: a tiny CNN with the shape of the classic LRP illustration, trained on MNIST
   without biases. The forward pass and LRP (z+ rule) run in the page, for whatever the reader draws on the input. */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.append(e); return e; };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = (t) => t * t * (3 - 2 * t);
  const fmt = (v) => (v > 0 && v < 0.01 ? v.toFixed(3) : Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));

  // layer l: 0 pixels (784), 1 first maps (3 x 196), 2 second maps (5 x 49), 3 dense (8), 4 features (5), 5 output (10)
  const GROUP = [784, 196, 49, 1, 1, 1];            // positions per drawn node
  const COUNTS = [1, 3, 5, 8, 5, 10];
  const node = (l, i) => (i / GROUP[l]) | 0;
  const NAME = (l, i) => ['the pixels', `first-layer map ${i + 1}`, `second-layer map ${i + 1}`, `dense unit ${i + 1}`, `feature ${i + 1}`, 'the output'][l];

  /* ---------- the network ---------- */
  const convFwd = (a, C, H, w, O) => {
    const Ho = ((H - 1) >> 1) + 1, z = new Float64Array(O * Ho * Ho), fwd = [...Array(C)].map(() => new Float64Array(O));
    for (let o = 0; o < O; o++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Ho; x++) {
      let s = 0;
      for (let c = 0; c < C; c++) for (let ky = 0; ky < 5; ky++) {
        const iy = y * 2 - 2 + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 5; kx++) {
          const ix = x * 2 - 2 + kx; if (ix < 0 || ix >= H) continue;
          const wv = w[((o * C + c) * 5 + ky) * 5 + kx], v = a[(c * H + iy) * H + ix] * wv;
          s += v; if (wv > 0) fwd[c][o] += v;
        }
      }
      z[(o * Ho + y) * Ho + x] = s;
    }
    return { z, fwd };
  };
  const denseFwd = (a, w, O, g = 1) => {
    const N = a.length, z = new Float64Array(O), fwd = [...Array(N / g)].map(() => new Float64Array(O));
    for (let o = 0; o < O; o++) for (let i = 0; i < N; i++) { const wv = w[o * N + i], v = a[i] * wv; z[o] += v; if (wv > 0) fwd[(i / g) | 0][o] += v; }
    return { z, fwd };
  };
  const relu = (z) => z.map((v) => Math.max(0, v));

  // z+ redistribution from layer l to l-1: fn(in, out, v) for every share v of R_out that goes to in
  const conv5 = (a, C, H, w, O) => (R, fn) => {
    const Ho = ((H - 1) >> 1) + 1;
    for (let o = 0; o < O; o++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Ho; x++) {
      const out = (o * Ho + y) * Ho + x, r = R[out]; if (!r) continue;
      let z = 0;
      const each = (cb) => { for (let c = 0; c < C; c++) for (let ky = 0; ky < 5; ky++) { const iy = y * 2 - 2 + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 5; kx++) { const ix = x * 2 - 2 + kx; if (ix < 0 || ix >= H) continue;
          const wv = w[((o * C + c) * 5 + ky) * 5 + kx]; if (wv > 0) cb((c * H + iy) * H + ix, wv); } } };
      each((i, wv) => { z += a[i] * wv; });
      if (z <= 0) continue;
      const s = r / z;
      each((i, wv) => fn(i, out, a[i] * wv * s));
    }
  };
  const dense = (a, w, O) => (R, fn) => {
    const N = a.length;
    for (let o = 0; o < O; o++) {
      if (!R[o]) continue;
      let z = 0; for (let i = 0; i < N; i++) { const wv = w[o * N + i]; if (wv > 0) z += a[i] * wv; }
      if (z <= 0) continue;
      for (let i = 0; i < N; i++) { const wv = w[o * N + i]; if (wv > 0) fn(i, o, a[i] * wv * R[o] / z); }
    }
  };

  /* ---------- colours and map images ---------- */
  const greyRamp = (t) => { const v = Math.round(255 - 205 * t); return [v, v, v + 2]; };
  const redRamp = (t) => {
    const stops = [[255, 255, 255], [240, 170, 160], [200, 65, 59], [110, 16, 14]];
    const x = clamp(t, 0, 1) * 3, i = Math.min(2, Math.floor(x)), f = x - i;
    return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  };
  const cv = document.createElement('canvas');
  const mapURL = (vals, n, ramp, max) => {
    cv.width = cv.height = n; const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) img.data.set([...ramp(max > 0 ? Math.max(0, vals[i]) / max : 0), 255], i * 4);
    ctx.putImageData(img, 0, 0); return cv.toDataURL();
  };
  const slice = (arr, l, i) => Array.from(arr.slice(i * GROUP[l], (i + 1) * GROUP[l]));
  // the image and the pixel map show the central 20 x 20 box only: the 4-pixel margin is black and gets no relevance
  const BOX = 20, PAD = 4;
  const inBox = (v) => Array.from({ length: BOX * BOX }, (_, q) => v[(PAD + ((q / BOX) | 0)) * 28 + PAD + (q % BOX)]);
  const SIZES = [28, 14, 7], MS = [124, 96, 62];
  const mapImage = (arr, l, i, ramp, max) => (l === 0 ? mapURL(inBox(arr), BOX, ramp, max) : mapURL(slice(arr, l, i), SIZES[l], ramp, max));
  const layerMax = (arr) => { let m = 1e-12; for (const v of arr) if (v > m) m = v; return m; };

  /* ---------- layout (viewBox 1200 x 490; the right half mirrors the left) ---------- */
  const W = 1200, H = 490, X = [66, 212, 330, 430, 515, 600], CY = 262;
  const ys = (l) => { const n = COUNTS[l], gap = [0, 128, 82, 46, 56, 40][l]; return Array.from({ length: n }, (_, i) => CY + (i - (n - 1) / 2) * gap); };
  const curve = (p, q) => { const mx = (p[0] + q[0]) / 2; return `M${p[0]},${p[1]} C${mx},${p[1]} ${mx},${q[1]} ${q[0]},${q[1]}`; };
  const square = (g, cx, cy, s, n) => {
    const x0 = cx - s / 2, y0 = cy - s / 2;
    S('rect', { class: 'lrpw-halo', x: x0 - 3, y: y0 - 3, width: s + 6, height: s + 6, rx: 2 }, g);
    const img = S('image', { x: x0, y: y0, width: s, height: s, preserveAspectRatio: 'none', style: 'image-rendering: pixelated' }, g);
    if (n <= 14) for (let i = 1; i < n; i++) {
      S('line', { class: 'lrpw-grid', x1: x0 + s * i / n, x2: x0 + s * i / n, y1: y0, y2: y0 + s }, g);
      S('line', { class: 'lrpw-grid', x1: x0, x2: x0 + s, y1: y0 + s * i / n, y2: y0 + s * i / n }, g);
    }
    S('rect', { class: 'lrpw-frame', x: x0, y: y0, width: s, height: s }, g);
    return { img, anchor: (side, off = 0) => [cx + side * s / 2, cy + off] };
  };
  const D = 1, T0 = 5 * D + 0.6, TOTAL = T0 + 5 * D + 0.6, SPEED = 4.2;

  const figure = async (root) => {
    const svg = root.querySelector('.lrpw-scene'), pad = root.querySelector('.lrpw-pad');
    const clearBtn = root.querySelector('.lrpw-clear'), readoutEl = root.querySelector('.lrpw-readout');
    let data;
    try { data = await fetch(root.dataset.src).then((r) => r.json()); } catch (e) { readoutEl.textContent = 'The network could not be loaded.'; return; }
    const [W0, W1, W2, W3, W4] = data.weights;

    const forward = (pixels) => {
      const a0 = Float64Array.from(pixels, (p) => p / 255);
      const c1 = convFwd(a0, 1, 28, W0, 3), a1 = relu(c1.z);
      const c2 = convFwd(a1, 3, 14, W1, 5), a2 = relu(c2.z);
      const d3 = denseFwd(a2, W2, 8, 49), a3 = relu(d3.z);
      const d4 = denseFwd(a3, W3, 5), a4 = relu(d4.z);
      const d5 = denseFwd(a4, W4, 10), logits = d5.z;
      const A = [a0, a1, a2, a3, a4, logits];
      const steps = [null, conv5(a0, 1, 28, W0, 3), conv5(a1, 3, 14, W1, 5), dense(a2, W2, 8), dense(a3, W3, 5), dense(a4, W4, 10)];
      const nodeAct = (l) => Array.from({ length: COUNTS[l] }, (_, i) => { let s = 0; for (let p = 0; p < GROUP[l]; p++) s += A[l][i * GROUP[l] + p]; return s / GROUP[l]; });
      return { A, steps, logits, act: [0, 1, 2, 3, 4].map(nodeAct), fwd: [c1.fwd, c2.fwd, d3.fwd, d4.fwd, d5.fwd] };
    };

    // LRP from the score of class cls, optionally kept to one node of one layer. Down: the relevance from the start
    // (only the node's share continues below it). Up: the part of every earlier flow that ends up in that node.
    const explain = (only) => {
      const keep = (l, p) => (only && only.l === l && node(l, p) !== only.i ? 0 : 1);
      const R = [], C = [];
      R[5] = new Float64Array(10); R[5][cls] = 1;
      for (let l = 5; l >= 1; l--) {
        const I = [], O = [], V = [], Rin = new Float64Array(net.A[l - 1].length);
        net.steps[l](R[l], (i, o, v) => { I.push(i); O.push(o); V.push(v); Rin[i] += v; });
        C[l] = { I, O, V };
        R[l - 1] = Rin.map((v, i) => v * keep(l - 1, i));
      }
      const disp = [R[0]], flows = [];
      let pass = new Float64Array(784).fill(1);      // the share of what arrives at a position that ends up in the result
      for (let l = 1; l <= 5; l++) {
        const { I, O, V } = C[l], t = new Float64Array(net.A[l].length), F = [...Array(COUNTS[l - 1])].map(() => new Float64Array(COUNTS[l]));
        for (let k = 0; k < V.length; k++) { const w = V[k] * pass[I[k]]; t[O[k]] += w; F[node(l - 1, I[k])][node(l, O[k])] += w; }
        disp[l] = t; flows[l - 1] = F;
        pass = t.map((v, p) => (R[l][p] > 0 ? keep(l, p) * v / R[l][p] : 0));
      }
      return { disp, flows };
    };
    // the forward flow from one node on: the share of each later unit's positive input that traces back to it
    const traceForward = (l0, i0) => {
      const share = [], part = [0, 1, 2, 3, 4, 5].map((l) => new Uint8Array(COUNTS[l]));
      part[l0][i0] = 1;
      let tr = net.A[l0].map((_, p) => (node(l0, p) === i0 ? 1 : 0));
      for (let l = l0 + 1; l <= 5; l++) {
        const prev = tr, a = net.A[l - 1];
        tr = new Float64Array(net.A[l].length);
        net.steps[l](new Float64Array(net.A[l].length).fill(1), (i, o, v) => { tr[o] += v * prev[i]; });
        share[l - 1] = Array.from({ length: COUNTS[l - 1] }, (_, j) => {
          let s = 0, t = 0; for (let p = j * GROUP[l - 1]; p < (j + 1) * GROUP[l - 1]; p++) { s += a[p] * prev[p]; t += a[p]; }
          return t > 0 ? s / t : 0;
        });
        for (let k = 0; k < COUNTS[l]; k++) {
          let s = 0; for (let p = k * GROUP[l]; p < (k + 1) * GROUP[l]; p++) s += tr[p];
          if (s > 1e-6) part[l][k] = 1;
        }
      }
      return { share, part };
    };

    /* ---------- state ---------- */
    // cls: the class being explained (the kept one, or the one under the pointer); pred: the network's prediction
    const state = { pixels: data.start, hover: null, classHover: null, t: 0, playing: false, played: false };
    let net, scene, cls, pinned, pred, full, views, probs;
    const fulls = new Map(), viewSets = new Map();

    /* ---------- geometry, once per digit ---------- */
    const build = () => {
      svg.replaceChildren();
      S('text', { class: 'lrpw-title', x: 300, y: 34, 'text-anchor': 'middle' }, svg).textContent = 'Prediction';
      S('text', { class: 'lrpw-title lrpw-title--rel', x: 900, y: 34, 'text-anchor': 'middle' }, svg).textContent = 'Explanation';
      S('line', { class: 'lrpw-divider', x1: 600, x2: 600, y1: 52, y2: 480 }, svg);
      S('text', { class: 'lrpw-hint', x: X[0], y: CY - MS[0] / 2 - 10, 'text-anchor': 'middle' }, svg).textContent = 'draw a digit here';
      const gE = S('g', {}, svg), gN = S('g', {}, svg);
      const L = { fwd: [], rel: [] }, fwdEdges = [], relEdges = [];
      for (const side of ['fwd', 'rel']) for (let l = 0; l < 6; l++) {
        if (l === 5 && side === 'rel') { L.rel[5] = L.fwd[5]; continue; }
        L[side][l] = ys(l).map((y, i) => {
          const x = side === 'fwd' ? X[l] : W - X[l], g = S('g', {}, gN), rec = { l, i, side, x, y, el: g };
          if (l <= 2) Object.assign(rec, square(g, x, y, MS[l], SIZES[l]));
          else {
            const r = l === 5 ? 11 : 10;
            S('circle', { class: 'lrpw-halo', cx: x, cy: y, r: r + 3.5 }, g);
            S('circle', { class: 'lrpw-node', cx: x, cy: y, r }, g);
            const fill = S('circle', { class: `lrpw-fill lrpw-fill--${side}`, cx: x, cy: y, r: 0 }, g);
            Object.assign(rec, { fill, r, anchor: (s2) => [x + s2 * r, y] });
            if (l === 5) S('text', { class: 'lrpw-digit', x, y: y + 4, 'text-anchor': 'middle' }, g).textContent = i;
          }
          return rec;
        });
      }
      const off = (rec, n, idx) => (rec.l <= 2 ? (idx - (n - 1) / 2) * Math.min(5, MS[rec.l] * 0.5 / n) : 0);
      for (let l = 0; l < 5; l++) for (let j = 0; j < COUNTS[l]; j++) for (let k = 0; k < COUNTS[l + 1]; k++) {
        const a = L.fwd[l][j], b = L.fwd[l + 1][k];
        const el = S('path', { class: 'lrpw-edge lrpw-edge--fwd', d: curve(a.anchor(1, off(a, COUNTS[l + 1], k)), b.anchor(-1, off(b, COUNTS[l], j))) }, gE);
        const dash = S('path', { class: 'lrpw-flow', d: el.getAttribute('d'), style: 'display:none' }, gE);
        fwdEdges.push({ el, dash, t0: l * D, t1: (l + 1) * D, l, j, k });
      }
      for (let l = 4; l >= 0; l--) for (let j = 0; j < COUNTS[l]; j++) for (let k = 0; k < COUNTS[l + 1]; k++) {
        const a = L.rel[l + 1][k], b = L.rel[l][j];
        const el = S('path', { class: 'lrpw-edge lrpw-edge--rel', d: curve(a.anchor(1, off(a, COUNTS[l], j)), b.anchor(-1, off(b, COUNTS[l + 1], k))) }, gE);
        const dash = S('path', { class: 'lrpw-flow', d: el.getAttribute('d'), style: 'display:none' }, gE);
        relEdges.push({ el, dash, t0: T0 + (4 - l) * D, t1: T0 + (5 - l) * D, l, j, k });
      }
      // hover: units and whole maps on both sides (not the image, the pixels or the output); a tap does the same
      for (const side of ['fwd', 'rel']) for (let l = 1; l <= 4; l++) L[side][l].forEach((rec) => {
        const h = { side, l, i: rec.i };
        rec.el.classList.add('lrpw-hoverable');
        rec.el.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') setHover(h); });
        rec.el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') setHover(null); });
        rec.el.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') { e.stopPropagation(); setHover(state.hover && key(state.hover) === key(h) ? null : h); } });
      });
      // the output: hover a class to preview its explanation, click to keep it
      L.fwd[5].forEach((rec) => {
        rec.el.classList.add('lrpw-hoverable');
        rec.el.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { state.classHover = rec.i; setClass(rec.i); } });
        rec.el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') { state.classHover = null; setClass(pinned); } });
        rec.el.addEventListener('click', () => { pinned = rec.i; setClass(pinned); });
      });
      [...fwdEdges, ...relEdges].forEach((e) => { e.len = e.el.getTotalLength(); e.el.style.strokeDasharray = `${e.len} ${e.len + 1}`; });
      scene = { L, fwdEdges, relEdges };

      // the fixed forward look: activations, and each edge's positive contribution as a share of what the receiving
      // unit gets, times how active that unit is
      L.fwd.forEach((layer, l) => layer.forEach((rec) => {
        if (l <= 2) rec.img.setAttribute('href', mapImage(net.A[l], l, rec.i, greyRamp, layerMax(net.A[l])));
        else if (l < 5) rec.fill.setAttribute('r', Math.max(0, (rec.r - 2.2) * Math.sqrt(clamp(net.act[l][rec.i] / Math.max(1e-9, ...net.act[l]), 0, 1))));
        else rec.fill.setAttribute('r', Math.max(0, (rec.r - 2.2) * Math.sqrt(probs[rec.i])));
      }));
      for (let l = 0; l < 5; l++) {
        const F = net.fwd[l];
        const tot = Array.from({ length: COUNTS[l + 1] }, (_, k) => F.reduce((s, row) => s + Math.max(0, row[k]), 0));
        const actK = l === 4 ? net.logits.map((v) => Math.max(0, v)) : net.act[l + 1], mx = Math.max(1e-9, ...actK);
        fwdEdges.filter((e) => e.l === l).forEach((e) => {
          const v = F[e.j][e.k];
          e.w0 = v > 0 ? 0.5 + 7 * (tot[e.k] > 0 ? v / tot[e.k] : 0) * clamp(actK[e.k] / mx, 0, 1) : 0.3;
        });
      }
    };

    /* ---------- views: computed once per hovered node and class, then kept ---------- */
    const key = (h) => (h ? `${h.side}|${h.l}|${h.i}` : 'none');
    const view = (h) => {
      const k = key(h);
      if (views.has(k)) return views.get(k);
      const v = { fwd: null, rel: {}, lit: { fwd: null } };
      let ex = full, boost = 1;
      if (h) {
        ex = explain({ l: h.l, i: h.i });
        let held = 0; for (let p = 0; p < GROUP[h.l]; p++) held += full.disp[h.l][h.i * GROUP[h.l] + p];
        v.held = held;
        boost = held > 0 && held < 0.1 ? Math.min(300, 0.4 / held) : 1;     // only small shares are scaled up
      }
      v.boost = boost;
      // a hover on the left shows the way from the output back to the node, a hover on the right the way on to the pixels
      const edgeOn = (e) => !h || (h.side === 'fwd' ? e.l >= h.l : e.l < h.l);
      const nodeOn = (l, i) => !h || (l === h.l ? i === h.i : h.side === 'fwd' ? l > h.l : l < h.l);
      v.rel.edges = scene.relEdges.map((e) => (edgeOn(e) ? ex.flows[e.l][e.j][e.k] * 26 * boost : 0));
      v.rel.urls = [0, 1, 2].map((l) => { const m = layerMax(ex.disp[l]); return scene.L.rel[l].map((rec) => mapImage(ex.disp[l], l, rec.i, redRamp, m)); });
      v.rel.fills = [3, 4].map((l) => scene.L.rel[l].map((rec) => clamp(ex.disp[l][rec.i] * boost, 0, 1)));
      v.lit.rel = nodeOn;
      if (h && h.side === 'fwd') {
        const tr = traceForward(h.l, h.i);
        v.fwd = scene.fwdEdges.map((e) => (e.l >= h.l ? e.w0 * tr.share[e.l][e.j] : 0));
        v.lit.fwd = (l, i) => !!tr.part[l][i];
      }
      v.pix = ex.disp[0].reduce((s, x) => s + x, 0);
      views.set(k, v);
      return v;
    };

    const paint = () => {
      const h = state.hover, v = view(h), { L, fwdEdges, relEdges } = scene;
      relEdges.forEach((e, n) => {
        const w = v.rel.edges[n];
        e.el.style.strokeWidth = w > 1e-6 ? Math.max(0.45, w) : 0;
        e.el.style.strokeOpacity = w < 0.8 ? 0.35 : 0.85;
        e.big = w > 2.2;
        e.dash.style.display = e.big ? '' : 'none';
        if (e.big) e.dash.style.strokeWidth = Math.min(2.2, w * 0.35);
      });
      fwdEdges.forEach((e, n) => {
        const w = v.fwd ? v.fwd[n] : e.w0, on = !v.fwd || w > 0.05;
        e.el.style.strokeWidth = on ? Math.max(0.4, w) : 0.3;
        e.el.style.strokeOpacity = on ? (w < 1 ? 0.45 : 0.9) : 0.12;
        e.big = on && w > 2.2;
        e.dash.style.display = e.big ? '' : 'none';
        if (e.big) e.dash.style.strokeWidth = Math.min(2.2, w * 0.35);
      });
      for (let l = 0; l <= 4; l++) L.rel[l].forEach((rec) => {
        if (l <= 2) rec.img.setAttribute('href', v.rel.urls[l][rec.i]);
        else rec.fill.setAttribute('r', Math.max(0, (rec.r - 2.2) * Math.sqrt(v.rel.fills[l - 3][rec.i])));
        rec.el.classList.toggle('lrpw-dim', !v.lit.rel(l, rec.i));
        rec.el.classList.toggle('lrpw-hover', !!(h && h.side === 'rel' && h.l === l && h.i === rec.i));
        rec.el.classList.toggle('lrpw-mirror', !!(h && h.side === 'fwd' && h.l === l && h.i === rec.i));
      });
      L.fwd[5].forEach((rec) => rec.el.classList.toggle('lrpw-explained', rec.i === cls));
      L.fwd.forEach((layer, l) => layer.forEach((rec) => {
        rec.el.classList.toggle('lrpw-dim', !!(v.lit.fwd && !v.lit.fwd(l, rec.i)));
        rec.el.classList.toggle('lrpw-hover', !!(h && h.side === 'fwd' && h.l === l && h.i === rec.i));
      }));
      timeSet(state.t);
      readout(v);
    };
    const setClass = (k) => {
      if (k !== cls) {
        cls = k;
        if (!fulls.has(k)) { fulls.set(k, explain(null)); viewSets.set(k, new Map()); }
        full = fulls.get(k); views = viewSets.get(k);
      }
      paint();
    };
    const setHover = (h) => { if (key(h) !== key(state.hover)) { state.hover = h; paint(); } };

    // the line under the figure: what is under the pointer, then what the flows show
    const readout = (v) => {
      const h = state.hover, k = state.classHover, cap = (t) => t[0].toUpperCase() + t.slice(1), parts = [];
      if (h) {
        const a = net.act[h.l][h.i];
        parts.push(`<b>${cap(NAME(h.l, h.i))}</b> has a mean activation of <b>${fmt(a)}</b> and holds <b>${fmt(v.held)}</b> of the relevance.${a <= 0 ? ' It is silent for this digit, so it gets no relevance.' : ''}`);
        parts.push(h.side === 'fwd' ? 'The grey flow shows what it feeds on the way to the output, and the red flow shows the share of the relevance that comes back to it.'
          : 'The red flow shows where its relevance goes from there, down to the pixels.');
        if (v.boost > 1) parts.push(`Red line widths are scaled up ${v.boost < 10 ? v.boost.toFixed(1) : Math.round(v.boost)} times for this small share.`);
      } else {
        if (k !== null) parts.push(`<b>Class ${k}</b> has a score of <b>${net.logits[k].toFixed(1)}</b> and a probability of <b>${(probs[k] * 100).toFixed(probs[k] < 0.01 ? 2 : 0)}%</b>.${k !== pinned ? ' Click to keep explaining it.' : ''}`);
        parts.push(`All of the relevance flows. The amount of 1.00 starts at the score of class ${cls}${cls !== pred ? ` (the network predicts ${pred})` : ''}, and ${fmt(v.pix)} of it reaches the pixels.`);
      }
      if (net.logits[cls] < 0) parts.push(`The score of class ${cls} is negative, so the red flow shows only the evidence for it, which the evidence against outweighs.`);
      readoutEl.innerHTML = parts.join(' ');
    };

    /* ---------- the animation, once, when the figure first comes into view ---------- */
    const timeSet = (t) => {
      state.t = clamp(t, 0, TOTAL);
      [...scene.fwdEdges, ...scene.relEdges].forEach((e) => {
        const p = ease(clamp((state.t - e.t0) / (e.t1 - e.t0), 0, 1));
        e.el.style.strokeDashoffset = e.len * (1 - p);
        if (e.big) e.dash.style.opacity = p >= 1 ? 1 : 0;
      });
      scene.L.fwd.forEach((layer, l) => layer.forEach((rec) => { (rec.img || rec.fill).style.opacity = clamp((state.t - l * D + (l ? 0 : 1)) / 0.45, 0, 1); }));
      scene.L.rel.forEach((layer, l) => { if (l < 5) layer.forEach((rec) => { (rec.img || rec.fill).style.opacity = clamp((state.t - (T0 + (5 - l) * D)) / 0.45, 0, 1); }); });
    };
    let last = 0;
    const tick = (now) => {
      if (!state.playing) return;
      timeSet(state.t + (now - last) / 1000 * SPEED); last = now;
      if (state.t >= TOTAL) { state.playing = false; return; }
      requestAnimationFrame(tick);
    };
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    new IntersectionObserver(([en]) => {
      svg.classList.toggle('lrpw-paused', !en.isIntersecting);      // the moving dashes only run on screen
      if (en.isIntersecting && !state.played) {
        state.played = true;
        if (reduced) timeSet(TOTAL);
        else { timeSet(0); state.playing = true; last = performance.now(); requestAnimationFrame(tick); }
      }
    }, { threshold: 0.4 }).observe(svg);

    const load = () => {
      state.hover = null; state.classHover = null; fulls.clear(); viewSets.clear();
      net = forward(state.pixels);
      pred = pinned = cls = net.logits.indexOf(Math.max(...net.logits));
      const m = Math.max(...net.logits), ex = net.logits.map((v) => Math.exp(v - m)), sum = ex.reduce((a, b) => a + b, 0);
      probs = ex.map((v) => v / sum);
      build();
      full = explain(null); views = new Map(); fulls.set(cls, full); viewSets.set(cls, views);
      paint();
      if (!state.playing) timeSet(state.played ? TOTAL : 0);
    };

    /* ---------- drawing on the input ---------- */
    // ink: a 280 x 280 buffer, white on black like MNIST, that fills the 20 x 20 digit box of the 28 x 28 input.
    // pad: a transparent canvas over the input square that shows the strokes while the pen is down.
    const pctx = pad.getContext('2d'), P = pad.width;
    const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const ink = canvas(P, P), ictx = ink.getContext('2d');
    const blank = () => { ictx.fillStyle = '#000'; ictx.fillRect(0, 0, P, P); };
    { // the pad and the clear button sit over the input square, in percentages of the scaled figure
      const x0 = X[0] - MS[0] / 2, y0 = CY - MS[0] / 2, s = MS[0];
      Object.assign(pad.style, { left: `${x0 / W * 100}%`, top: `${y0 / H * 100}%`, width: `${s / W * 100}%`, height: `${s / H * 100}%` });
      Object.assign(clearBtn.style, { left: `${X[0] / W * 100}%`, top: `calc(${(y0 + s) / H * 100}% + 8px)` });
    }
    blank();
    { const c = canvas(BOX, BOX), x = c.getContext('2d'), img = x.createImageData(BOX, BOX);
      inBox(state.pixels).forEach((v, k) => img.data.set([v, v, v, 255], k * 4)); x.putImageData(img, 0, 0);
      ictx.imageSmoothingEnabled = true; ictx.drawImage(c, 0, 0, P, P); }
    const toPixels = () => {
      let c = ink;
      while (c.width > 56) {                           // halve in steps, so the strokes stay smooth
        const n = canvas(Math.ceil(c.width / 2), Math.ceil(c.height / 2)), nx = n.getContext('2d');
        nx.imageSmoothingQuality = 'high'; nx.drawImage(c, 0, 0, n.width, n.height); c = n;
      }
      const o = canvas(28, 28), ox = o.getContext('2d');
      ox.fillStyle = '#000'; ox.fillRect(0, 0, 28, 28);
      ox.imageSmoothingQuality = 'high'; ox.drawImage(c, PAD, PAD, BOX, BOX);
      const d = ox.getImageData(0, 0, 28, 28).data;
      return Array.from({ length: 784 }, (_, i) => d[i * 4]);
    };
    let drawing = false, lastPt = null;
    const at = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * P, (e.clientY - r.top) / r.height * P]; };
    const stroke = (a, b) => {
      for (const [ctx, colour] of [[ictx, '#fff'], [pctx, '#26302b']]) {
        ctx.strokeStyle = colour; ctx.lineWidth = 30; ctx.lineCap = ctx.lineJoin = 'round';
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      }
    };
    pad.addEventListener('pointerdown', (e) => { drawing = true; pad.setPointerCapture(e.pointerId); lastPt = at(e); stroke(lastPt, lastPt); });
    pad.addEventListener('pointermove', (e) => { if (!drawing) return; const p = at(e); stroke(lastPt, p); lastPt = p; });
    const finish = () => {
      if (!drawing) return;
      drawing = false;
      state.pixels = toPixels();
      load();
      pctx.clearRect(0, 0, P, P);
    };
    pad.addEventListener('pointerup', finish);
    pad.addEventListener('pointercancel', finish);
    clearBtn.addEventListener('click', () => {
      blank(); state.pixels = new Array(784).fill(0); load();
      readoutEl.textContent = 'Draw a digit on the input, filling the square.';
    });
    load();
  };

  const init = () => document.querySelectorAll('[data-lrp-water]').forEach((el) => {
    if (el.dataset.ready) return;
    el.dataset.ready = 'true';
    figure(el);
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

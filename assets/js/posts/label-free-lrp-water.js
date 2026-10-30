/* The water figure of the LRP post: a tiny CNN with the shape of the classic LRP illustration, trained on MNIST
   without biases. The forward pass and the way back (z+, LRP-0, gradient or guided backpropagation) run in the page, for
   whatever the reader draws on the input. */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.append(e); return e; };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (v) => (v > 0 && v < 0.01 ? v.toFixed(3) : Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
  const fmtTot = (v) => (Math.abs(v) < 0.005 ? '0.00' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)).replace('-', '−');

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
    for (let o = 0; o < O; o++) for (let i = 0; i < N; i++) { const v = a[i] * w[o * N + i]; z[o] += v; if (v > 0) fwd[(i / g) | 0][o] += v; }
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
      let z = 0; for (let i = 0; i < N; i++) { const c = a[i] * w[o * N + i]; if (c > 0) z += c; }
      if (z <= 0) continue;
      for (let i = 0; i < N; i++) { const c = a[i] * w[o * N + i]; if (c > 0) fn(i, o, c * R[o] / z); }
    }
  };
  const conv0 = (a, C, H, w, O) => (R, fn) => {
    const Ho = ((H - 1) >> 1) + 1;
    for (let o = 0; o < O; o++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Ho; x++) {
      const out = (o * Ho + y) * Ho + x, r = R[out]; if (!r) continue;
      let z = 0;
      const each = (cb) => { for (let c = 0; c < C; c++) for (let ky = 0; ky < 5; ky++) { const iy = y * 2 - 2 + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 5; kx++) { const ix = x * 2 - 2 + kx; if (ix < 0 || ix >= H) continue;
          cb((c * H + iy) * H + ix, w[((o * C + c) * 5 + ky) * 5 + kx]); } } };
      each((i, wv) => { z += a[i] * wv; });
      if (Math.abs(z) < 1e-9) continue;
      const s = r / z;
      each((i, wv) => fn(i, out, a[i] * wv * s));
    }
  };
  const dense0 = (a, w, O) => (R, fn) => {
    const N = a.length;
    for (let o = 0; o < O; o++) {
      if (!R[o]) continue;
      let z = 0; for (let i = 0; i < N; i++) z += a[i] * w[o * N + i];
      if (Math.abs(z) < 1e-9) continue;
      for (let i = 0; i < N; i++) fn(i, o, a[i] * w[o * N + i] * R[o] / z);
    }
  };
  const gconv5 = (C, H, w, O, z, guided) => (R, fn) => {
    const Ho = ((H - 1) >> 1) + 1;
    for (let o = 0; o < O; o++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Ho; x++) {
      const out = (o * Ho + y) * Ho + x, r = R[out]; if (!r || (z && (z[out] <= 0 || (guided && r < 0)))) continue;
      for (let c = 0; c < C; c++) for (let ky = 0; ky < 5; ky++) { const iy = y * 2 - 2 + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 5; kx++) { const ix = x * 2 - 2 + kx; if (ix < 0 || ix >= H) continue;
          fn((c * H + iy) * H + ix, out, w[((o * C + c) * 5 + ky) * 5 + kx] * r); } }
    }
  };
  const gdense = (N, w, O, z, guided) => (R, fn) => {
    for (let o = 0; o < O; o++) {
      if (!R[o] || (z && (z[o] <= 0 || (guided && R[o] < 0)))) continue;
      for (let i = 0; i < N; i++) fn(i, o, w[o * N + i] * R[o]);
    }
  };

  const convOp = (C, H, w, O) => ({
    forward: (a) => convFwd(a, C, H, w, O),
    step: (mode, a, z) => (mode === 'lrp0' ? conv0(a, C, H, w, O) : mode === 'rules' ? conv5(a, C, H, w, O) : gconv5(C, H, w, O, z, mode === 'guided')),
  });
  const denseOp = (w, O, g = 1) => ({
    forward: (a) => denseFwd(a, w, O, g),
    step: (mode, a, z) => (mode === 'lrp0' ? dense0(a, w, O) : mode === 'rules' ? dense(a, w, O) : gdense(a.length, w, O, z, mode === 'guided')),
  });

  const digitNet = ([W0, W1, W2, W3, W4]) => ({
    layers: [
      { kind: 'image', count: 1, side: 28, ms: 124, x: 66, gap: 0, name: () => 'the pixels' },
      { kind: 'map', count: 3, side: 14, ms: 96, x: 212, gap: 128, relu: true, name: (i) => `first-layer map ${i + 1}` },
      { kind: 'map', count: 5, side: 7, ms: 62, x: 330, gap: 82, relu: true, name: (i) => `second-layer map ${i + 1}` },
      { kind: 'unit', count: 8, x: 430, gap: 46, relu: true, name: (i) => `dense unit ${i + 1}` },
      { kind: 'unit', count: 5, x: 515, gap: 56, relu: true, name: (i) => `feature ${i + 1}` },
      { kind: 'output', count: 10, x: 600, gap: 40, name: () => 'the output' },
    ],
    links: [
      { from: 0, to: 1, op: convOp(1, 28, W0, 3) },
      { from: 1, to: 2, op: convOp(3, 14, W1, 5) },
      { from: 2, to: 3, op: denseOp(W2, 8, 49) },
      { from: 3, to: 4, op: denseOp(W3, 5) },
      { from: 4, to: 5, op: denseOp(W4, 10) },
    ],
  });

  /* ---------- colours and map images ---------- */
  const greyRamp = (t) => { const v = Math.round(255 - 205 * t); return [v, v, v + 2]; };
  const redRamp = (t) => {
    const stops = [[255, 255, 255], [240, 170, 160], [200, 65, 59], [110, 16, 14]];
    const x = clamp(t, 0, 1) * 3, i = Math.min(2, Math.floor(x)), f = x - i;
    return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  };
  const blueRamp = (t) => {
    const stops = [[255, 255, 255], [172, 196, 238], [52, 108, 196], [16, 40, 104]];
    const x = clamp(t, 0, 1) * 3, i = Math.min(2, Math.floor(x)), f = x - i;
    return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  };
  const cv = document.createElement('canvas');
  const mapURL = (vals, n, ramp, max) => {
    cv.width = cv.height = n; const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) img.data.set([...ramp(max > 0 ? Math.max(0, vals[i]) / max : 0), 255], i * 4);
    ctx.putImageData(img, 0, 0); return cv.toDataURL();
  };
  const signedURL = (vals, n, max) => {
    cv.width = cv.height = n; const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) { const t = max > 0 ? vals[i] / max : 0; img.data.set([...(t >= 0 ? redRamp(t) : blueRamp(-t)), 255], i * 4); }
    ctx.putImageData(img, 0, 0); return cv.toDataURL();
  };
  const slice = (arr, ly, i) => Array.from(arr.slice(i * ly.group, (i + 1) * ly.group));
  // the image and the pixel map show the central 20 x 20 box only: the 4-pixel margin is black and gets no relevance
  const BOX = 20, PAD = 4;
  const inBox = (v) => Array.from({ length: BOX * BOX }, (_, q) => v[(PAD + ((q / BOX) | 0)) * 28 + PAD + (q % BOX)]);
  const mapImage = (arr, ly, i, ramp, max) => (ly.kind === 'image' ? mapURL(inBox(arr), BOX, ramp, max) : mapURL(slice(arr, ly, i), ly.side, ramp, max));
  const signedImage = (arr, ly, i, max) => (ly.kind === 'image' ? signedURL(inBox(arr), BOX, max) : signedURL(slice(arr, ly, i), ly.side, max));
  const layerMax = (arr) => { let m = 1e-12; for (const v of arr) if (v > m) m = v; return m; };
  const absMax = (arr) => { let m = 1e-12; for (const v of arr) if (Math.abs(v) > m) m = Math.abs(v); return m; };
  const sum = (arr) => arr.reduce((s, x) => s + x, 0);

  /* ---------- layout (viewBox 1200 x 490; the right half mirrors the left) ---------- */
  const W = 1200, H = 490, CY = 262;
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
  const swap = (e, a, b) => { const t = e.getAttribute(a); e.setAttribute(a, e.getAttribute(b)); e.setAttribute(b, t); };
  const turn = (svg) => svg.querySelectorAll('circle, text, rect, image, line, path').forEach((e) => {
    if (e.tagName === 'path') e.setAttribute('d', e.getAttribute('d').replace(/(-?[\d.]+),(-?[\d.]+)/g, '$2,$1'));
    else if (e.tagName === 'circle') swap(e, 'cx', 'cy');
    else if (e.tagName === 'line') { swap(e, 'x1', 'y1'); swap(e, 'x2', 'y2'); }
    else { swap(e, 'x', 'y'); if (e.hasAttribute('width')) swap(e, 'width', 'height'); }
  });
  const narrow = matchMedia('(max-width: 640px)');
  const weights = new Map();

  const figure = async (root) => {
    const svg = root.querySelector('.lrpw-scene'), pad = root.querySelector('.lrpw-pad');
    const clearBtn = root.querySelector('.lrpw-clear'), readoutEl = root.querySelector('.lrpw-readout');
    let data;
    try {
      if (!weights.has(root.dataset.src)) weights.set(root.dataset.src, fetch(root.dataset.src).then((r) => r.json()));
      data = await weights.get(root.dataset.src);
    } catch (e) { readoutEl.textContent = 'The network could not be loaded.'; return; }
    const spec = digitNet(data.weights), LY = spec.layers, LK = spec.links, L = LY.length - 1;
    LY.forEach((ly) => {
      ly.group = ly.side ? ly.side * ly.side : 1;
      ly.ys = Array.from({ length: ly.count }, (_, i) => CY + (i - (ly.count - 1) / 2) * ly.gap);
    });
    LK.forEach((lk, n) => { lk.n = n; });
    const into = LY.map((_, l) => LK.filter((lk) => lk.to === l));
    const node = (l, p) => (p / LY[l].group) | 0;
    const isMap = (l) => LY[l].kind === 'image' || LY[l].kind === 'map';

    const forward = (pixels) => {
      const A = [Float64Array.from(pixels, (p) => p / 255)], Z = [null], F = [];
      for (let l = 1; l <= L; l++) {
        let z = null;
        for (const lk of into[l]) { const r = lk.op.forward(A[lk.from]); F[lk.n] = r.fwd; z = z ? z.map((v, p) => v + r.z[p]) : r.z; }
        Z[l] = z; A[l] = LY[l].relu ? relu(z) : z;
      }
      const act = LY.slice(0, L).map((ly, l) => Array.from({ length: ly.count }, (_, i) => { let s = 0; for (let p = 0; p < ly.group; p++) s += A[l][i * ly.group + p]; return s / ly.group; }));
      const cache = {};
      const stepsFor = (mode) => (cache[mode] ??= LK.map((lk) => lk.op.step(mode, A[lk.from], LY[lk.to].relu ? Z[lk.to] : null)));
      return { A, Z, F, act, logits: A[L], stepsFor };
    };

    const zeros = () => LY.map((_, l) => new Float64Array(net.A[l].length));
    const backward = (R, top, steps) => {
      const C = [];
      for (let l = top; l >= 1; l--) for (const lk of into[l]) {
        const I = [], O = [], V = [], Rs = R[lk.from];
        steps[lk.n](R[l], (i, o, v) => { I.push(i); O.push(o); V.push(v); Rs[i] += v; });
        C[lk.n] = { I, O, V };
      }
      return C;
    };
    const flowMatrix = (lk, C, weight) => {
      const F = [...Array(LY[lk.from].count)].map(() => new Float64Array(LY[lk.to].count));
      if (C) { const { I, O, V } = C; for (let k = 0; k < V.length; k++) F[node(lk.from, I[k])][node(lk.to, O[k])] += V[k] * (weight ? weight[I[k]] : 1); }
      return F;
    };
    // LRP from the score of class cls, optionally kept to one node of one layer. Down: the relevance from the start
    // (only the node's share continues below it). Up: the part of every earlier flow that ends up in that node.
    const explain = (only) => {
      const steps = net.stepsFor(state.mode), R = zeros();
      R[L][cls] = startOf(cls);
      const C = backward(R, L, steps);
      if (!only) return { disp: R, flows: LK.map((lk) => flowMatrix(lk, C[lk.n])) };
      const keep = (p) => (node(only.l, p) === only.i ? 1 : 0);
      const R2 = zeros();
      R2[only.l] = R[only.l].map((v, p) => v * keep(p));
      const C2 = backward(R2, only.l, steps);
      const disp = R2.slice(), flows = [], pass = [];
      pass[only.l] = R[only.l].map((_, p) => keep(p));
      for (let l = only.l + 1; l <= L; l++) {
        const t = new Float64Array(net.A[l].length);
        for (const lk of into[l]) {
          const w = pass[lk.from]; if (!w) continue;
          const { I, O, V } = C[lk.n]; for (let k = 0; k < V.length; k++) t[O[k]] += V[k] * w[I[k]];
        }
        disp[l] = t;
        pass[l] = t.map((v, p) => (Math.abs(R[l][p]) > 1e-12 ? v / R[l][p] : 0));
      }
      LK.forEach((lk) => { flows[lk.n] = lk.to <= only.l ? flowMatrix(lk, C2[lk.n]) : flowMatrix(lk, pass[lk.from] ? C[lk.n] : null, pass[lk.from]); });
      return { disp, flows };
    };
    // the forward flow from one node on: the share of each later unit's positive input that traces back to it
    const traceForward = (l0, i0) => {
      const steps = net.stepsFor('rules'), share = [], part = LY.map((ly) => new Uint8Array(ly.count)), tr = [];
      part[l0][i0] = 1;
      tr[l0] = net.A[l0].map((_, p) => (node(l0, p) === i0 ? 1 : 0));
      for (let l = l0 + 1; l <= L; l++) {
        tr[l] = new Float64Array(net.A[l].length);
        for (const lk of into[l]) {
          const prev = tr[lk.from]; if (!prev) continue;
          const a = net.A[lk.from], ly = LY[lk.from];
          steps[lk.n](new Float64Array(net.A[l].length).fill(1), (i, o, v) => { tr[l][o] += v * prev[i]; });
          share[lk.n] = Array.from({ length: ly.count }, (_, j) => {
            let s = 0, t = 0; for (let p = j * ly.group; p < (j + 1) * ly.group; p++) { s += a[p] * prev[p]; t += a[p]; }
            return t > 0 ? s / t : 0;
          });
        }
        for (let k = 0; k < LY[l].count; k++) {
          let s = 0; for (let p = k * LY[l].group; p < (k + 1) * LY[l].group; p++) s += tr[l][p];
          if (s > 1e-6) part[l][k] = 1;
        }
      }
      return { share, part };
    };

    /* ---------- state ---------- */
    // cls: the class being explained (the one under the pointer, or else the prediction); pred: the network's prediction
    const state = { pixels: data.start, hover: null, classHover: null, method: root.querySelector('.lrpw-mode[aria-pressed=true]')?.dataset.mode || 'lrp', sign: root.querySelector('.lrpw-sign[aria-pressed=true]')?.dataset.sign || 'pos' };
    const isLRP = () => state.mode === 'rules' || state.mode === 'lrp0';
    const startOf = (k) => (!isLRP() || net.logits[k] >= 0 ? 1 : state.mode === 'lrp0' ? -1 : 0);
    const modeOf = () => (state.method === 'gradient' ? (state.sign === 'pos' ? 'guided' : 'gradient') : (state.sign === 'pos' ? 'rules' : 'lrp0'));
    state.mode = modeOf();
    const leaveTimers = {};
    let touch = false;
    svg.addEventListener('pointerdown', (e) => { touch = e.pointerType !== 'mouse'; });
    const enterSoon = (kind) => { clearTimeout(leaveTimers[kind]); };
    const leaveSoon = (kind, fn) => { clearTimeout(leaveTimers[kind]); leaveTimers[kind] = setTimeout(fn, 160); };
    let net, scene, cls, pred, full, views, probs;
    const fulls = new Map(), viewSets = new Map();

    /* ---------- geometry, once per digit ---------- */
    const build = () => {
      svg.replaceChildren();
      const titles = [S('text', { class: 'lrpw-title', x: 300, y: 34, 'text-anchor': 'middle' }, svg), S('text', { class: 'lrpw-title lrpw-title--rel', x: 900, y: 34, 'text-anchor': 'middle' }, svg)];
      titles[0].textContent = 'Prediction';
      titles[1].textContent = 'Explanation';
      const totalLabel = S('text', { class: 'lrpw-total-label', x: W - LY[L - 1].x - 30, y: 474, 'text-anchor': 'end' }, svg);
      totalLabel.textContent = 'total';
      const totals = LY.slice(0, L).map((ly) => S('text', { class: 'lrpw-total', x: W - ly.x, y: 474, 'text-anchor': 'middle' }, svg));
      const hint = S('text', { class: 'lrpw-hint', x: LY[0].x, y: CY - LY[0].ms / 2 - 10, 'text-anchor': 'middle' }, svg);
      hint.textContent = 'draw a digit here';
      const gE = S('g', {}, svg), gN = S('g', {}, svg), gH = S('g', {}, svg);
      const nodes = { fwd: [], rel: [] }, fwdEdges = [], relEdges = [];
      for (const side of ['fwd', 'rel']) LY.forEach((ly, l) => {
        if (l === L && side === 'rel') { nodes.rel[L] = nodes.fwd[L]; return; }
        nodes[side][l] = ly.ys.map((y, i) => {
          const x = side === 'fwd' ? ly.x : W - ly.x, g = S('g', {}, gN), rec = { l, i, side, x, y, el: g };
          if (isMap(l)) Object.assign(rec, square(g, x, y, ly.ms, ly.side));
          else {
            const r = ly.kind === 'output' ? 11 : 10;
            S('circle', { class: 'lrpw-halo', cx: x, cy: y, r: r + 3.5 }, g);
            S('circle', { class: 'lrpw-node', cx: x, cy: y, r }, g);
            Object.assign(rec, { r, anchor: () => [x, y] });
            if (ly.kind === 'output') S('text', { class: 'lrpw-digit', x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, g).textContent = i;
            else rec.fill = S('circle', { class: `lrpw-fill lrpw-fill--${side}`, cx: x, cy: y, r: 0 }, g);
          }
          return rec;
        });
      });
      const off = (rec, n, idx) => (isMap(rec.l) ? (idx - (n - 1) / 2) * Math.min(5, LY[rec.l].ms * 0.5 / n) : 0);
      const pair = (side, lk, j, k) => {
        const a = nodes[side][side === 'fwd' ? lk.from : lk.to][side === 'fwd' ? j : k], b = nodes[side][side === 'fwd' ? lk.to : lk.from][side === 'fwd' ? k : j];
        const na = side === 'fwd' ? LY[lk.to].count : LY[lk.from].count, nb = side === 'fwd' ? LY[lk.from].count : LY[lk.to].count;
        return curve(a.anchor(1, off(a, na, side === 'fwd' ? k : j)), b.anchor(-1, off(b, nb, side === 'fwd' ? j : k)));
      };
      const edge = (side, lk, j, k) => {
        const el = S('path', { class: `lrpw-edge lrpw-edge--${side}`, d: pair(side, lk, j, k) }, gE);
        return { el, n: lk.n, j, k };
      };
      const pairs = (lk, fn) => { for (let j = 0; j < LY[lk.from].count; j++) for (let k = 0; k < LY[lk.to].count; k++) fn(j, k); };
      LK.forEach((lk) => pairs(lk, (j, k) => fwdEdges.push(edge('fwd', lk, j, k))));
      [...LK].reverse().forEach((lk) => pairs(lk, (j, k) => relEdges.push(edge('rel', lk, j, k))));
      // hover: units and whole maps on both sides (not the image, the pixels or the output); a tap does the same
      const half = (l) => (isMap(l) ? LY[l].ms / 2 + 12 : LY[l].kind === 'output' ? 20 : 26);
      const hit = (layer, l, rec) => {
        const k = layer.indexOf(rec), y = rec.y;
        const up = k > 0 ? (layer[k - 1].y + y) / 2 : y - (layer.length > 1 ? (layer[1].y - layer[0].y) / 2 : half(l));
        const dn = k < layer.length - 1 ? (layer[k + 1].y + y) / 2 : y + (layer.length > 1 ? (y - layer[k - 1].y) / 2 : half(l));
        return S('rect', { class: 'lrpw-hit', x: rec.x - half(l), y: up, width: 2 * half(l), height: dn - up }, gH);
      };
      for (const side of ['fwd', 'rel']) for (let l = 1; l < L; l++) nodes[side][l].forEach((rec) => {
        const h = { side, l, i: rec.i }, el = hit(nodes[side][l], l, rec);
        el.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { enterSoon('unit'); setHover(h); } });
        el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') leaveSoon('unit', () => setHover(null)); });
        el.addEventListener('click', () => { if (touch) setHover(state.hover && key(state.hover) === key(h) ? null : h); });
      });
      // the output: hover a class to explain it instead of the prediction; a tap does the same
      const showClass = (k) => { state.classHover = k; setClass(k ?? pred); };
      nodes.fwd[L].forEach((rec) => {
        const el = hit(nodes.fwd[L], L, rec);
        el.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') { enterSoon('class'); showClass(rec.i); } });
        el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') leaveSoon('class', () => showClass(null)); });
        el.addEventListener('click', () => { if (touch) showClass(state.classHover === rec.i ? null : rec.i); });
      });
      scene = { nodes, fwdEdges, relEdges, totals };
      // the pad and the clear button sit over the input square, in percentages of the scaled figure
      const tall = narrow.matches, x0 = LY[0].x - LY[0].ms / 2, y0 = CY - LY[0].ms / 2, s = LY[0].ms;
      const VW = H + 20, TOP = 24, VH = W + TOP;
      svg.setAttribute('viewBox', tall ? `0 ${-TOP} ${VW} ${VH}` : `0 0 ${W} ${H}`);
      if (tall) {
        turn(svg);
        [...totals, totalLabel].forEach((t) => t.setAttribute('x', H));
        titles.forEach((t) => t.setAttribute('transform', `rotate(-90 ${t.getAttribute('x')} ${t.getAttribute('y')})`));
        Object.assign(pad.style, { left: `${y0 / VW * 100}%`, top: `${(x0 + TOP) / VH * 100}%`, width: `${s / VW * 100}%`, height: `${s / VH * 100}%` });
        Object.assign(clearBtn.style, { left: `${(y0 - 60) / VW * 100}%`, top: `calc(${(LY[0].x + TOP) / VH * 100}% - 11px)` });
        hint.setAttribute('x', CY);
        hint.setAttribute('y', x0 - 8);
        totalLabel.setAttribute('text-anchor', 'middle');
      } else {
        Object.assign(pad.style, { left: `${x0 / W * 100}%`, top: `${y0 / H * 100}%`, width: `${s / W * 100}%`, height: `${s / H * 100}%` });
        Object.assign(clearBtn.style, { left: `${LY[0].x / W * 100}%`, top: `calc(${(y0 + s) / H * 100}% + 8px)` });
      }

      // the fixed forward look: activations, and each edge's positive contribution as a share of what the receiving
      // unit gets, times how active that unit is
      nodes.fwd.forEach((layer, l) => layer.forEach((rec) => {
        if (isMap(l)) rec.img.setAttribute('href', mapImage(net.A[l], LY[l], rec.i, greyRamp, layerMax(net.A[l])));
        else if (l < L) rec.fill.setAttribute('r', Math.max(0, (rec.r - 2.2) * Math.sqrt(clamp(net.act[l][rec.i] / Math.max(1e-9, ...net.act[l]), 0, 1))));
      }));
      LK.forEach((lk) => {
        const F = net.F[lk.n];
        const tot = Array.from({ length: LY[lk.to].count }, (_, k) => F.reduce((s, row) => s + Math.max(0, row[k]), 0));
        const actK = lk.to === L ? Array.from(net.logits, (v) => Math.max(0, v)) : net.act[lk.to], mx = Math.max(1e-9, ...actK);
        fwdEdges.filter((e) => e.n === lk.n).forEach((e) => {
          const v = F[e.j][e.k];
          e.w0 = v > 0 ? 0.5 + 7 * (tot[e.k] > 0 ? v / tot[e.k] : 0) * clamp(actK[e.k] / mx, 0, 1) : 0.3;
        });
      });
    };

    /* ---------- views: computed once per hovered node and class, then kept ---------- */
    const key = (h) => `${state.mode}|${h ? `${h.side}|${h.l}|${h.i}` : 'none'}`;
    const view = (h) => {
      const k = key(h);
      if (views.has(k)) return views.get(k);
      const v = { fwd: null, rel: {}, lit: { fwd: null } };
      let ex = full, boost = 1;
      if (h) {
        ex = explain({ l: h.l, i: h.i });
        const g = LY[h.l].group;
        let held = 0; for (let p = 0; p < g; p++) held += full.disp[h.l][h.i * g + p];
        v.held = held;
        boost = Math.abs(held) > 0 && Math.abs(held) < 0.1 ? Math.min(300, 0.4 / Math.abs(held)) : 1;     // only small shares are scaled up
      }
      v.boost = boost;
      // a hover on the left shows the round trip through the node, a hover on the right the way on to the pixels
      const edgeOn = (e) => !h || h.side === 'fwd' || LK[e.n].to <= h.l;
      const nodeOn = (l, i) => !h || (l === h.l ? i === h.i : h.side === 'fwd' || l < h.l);
      v.rel.edges = scene.relEdges.map((e) => (edgeOn(e) ? ex.flows[e.n][e.j][e.k] * 26 * boost : 0));
      v.rel.urls = [];
      v.rel.fills = [];
      LY.slice(0, L).forEach((ly, l) => {
        const d = ex.disp[l];
        if (isMap(l)) {
          const m = absMax(ex.disp[l]);
          v.rel.urls[l] = scene.nodes.rel[l].map((rec) => signedImage(d, ly, rec.i, m));
        } else v.rel.fills[l] = scene.nodes.rel[l].map((rec) => clamp(d[rec.i] * boost, -1, 1));
      });
      v.tot = LY.slice(0, L).map((_, l) => sum(ex.disp[l]));
      v.lit.rel = nodeOn;
      if (h && h.side === 'fwd') {
        const tr = traceForward(h.l, h.i);
        v.fwd = scene.fwdEdges.map((e) => (tr.share[e.n] ? e.w0 * tr.share[e.n][e.j] : 0));
        v.lit.fwd = (l, i) => !!tr.part[l][i];
      }
      v.pix = sum(ex.disp[0]);
      views.set(k, v);
      return v;
    };

    const paint = () => {
      const h = state.hover, v = view(h), { nodes, fwdEdges, relEdges } = scene;
      relEdges.forEach((e, n) => {
        const s = v.rel.edges[n], w = Math.abs(s);
        e.el.style.strokeWidth = w > 1e-6 ? clamp(w, 0.45, 14) : 0;
        e.el.style.strokeOpacity = w < 0.8 ? 0.35 : 0.85;
        e.el.classList.toggle('lrpw-neg', s < 0);
      });
      fwdEdges.forEach((e, n) => {
        const w = v.fwd ? v.fwd[n] : e.w0, on = !v.fwd || w > 0.05;
        e.el.style.strokeWidth = on ? Math.max(0.4, w) : 0.3;
        e.el.style.strokeOpacity = on ? (w < 1 ? 0.45 : 0.9) : 0.12;
      });
      for (let l = 0; l < L; l++) nodes.rel[l].forEach((rec) => {
        if (isMap(l)) rec.img.setAttribute('href', v.rel.urls[l][rec.i]);
        else {
          const f = v.rel.fills[l][rec.i];
          rec.fill.setAttribute('r', Math.max(0, (rec.r - 2.2) * Math.sqrt(Math.abs(f))));
          rec.fill.classList.toggle('lrpw-neg', f < 0);
        }
        rec.el.classList.toggle('lrpw-dim', !v.lit.rel(l, rec.i));
        rec.el.classList.toggle('lrpw-hover', !!(h && h.side === 'rel' && h.l === l && h.i === rec.i));
        rec.el.classList.toggle('lrpw-mirror', !!(h && h.side === 'fwd' && h.l === l && h.i === rec.i));
      });
      nodes.fwd[L].forEach((rec) => { rec.el.classList.toggle('lrpw-explained', rec.i === cls); rec.el.classList.toggle('lrpw-hover', rec.i === state.classHover); });
      scene.totals.forEach((t, l) => { t.textContent = fmtTot(v.tot[l]); });
      nodes.fwd.forEach((layer, l) => layer.forEach((rec) => {
        rec.el.classList.toggle('lrpw-dim', !!(v.lit.fwd && !v.lit.fwd(l, rec.i)));
        rec.el.classList.toggle('lrpw-hover', !!(h && h.side === 'fwd' && h.l === l && h.i === rec.i));
      }));
      readout(v);
    };
    const refresh = () => {
      const fk = `${state.mode}|${cls}`;
      if (!fulls.has(fk)) { fulls.set(fk, explain(null)); viewSets.set(fk, new Map()); }
      full = fulls.get(fk); views = viewSets.get(fk);
    };
    const setClass = (k) => { cls = k; refresh(); paint(); };
    const setHover = (h) => { if (key(h) !== key(state.hover)) { state.hover = h; paint(); } };
    for (const [sel, field] of [['.lrpw-mode', 'method'], ['.lrpw-sign', 'sign']]) root.querySelectorAll(sel).forEach((b) => b.addEventListener('click', () => {
      state[field] = b.dataset[field === 'method' ? 'mode' : 'sign']; state.mode = modeOf(); state.hover = null;
      root.querySelectorAll(sel).forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
      refresh(); paint();
    }));

    // the line under the figure: what is under the pointer, then what the flows show
    const readout = (v) => {
      const h = state.hover, k = state.classHover, cap = (t) => t[0].toUpperCase() + t.slice(1), parts = [];
      if (h) {
        const a = net.act[h.l][h.i];
        parts.push(`<b>${cap(LY[h.l].name(h.i))}</b> has a mean activation of <b>${fmt(a)}</b> and holds <b>${fmt(v.held)}</b> of the relevance.${a <= 0 && LY[h.l].relu ? ' It is silent for this digit, so it gets no relevance.' : ''}`);
        const flow = state.mode !== 'rules' ? 'red and blue flow' : 'red flow';
        parts.push(h.side === 'fwd' ? `The grey flow shows what it feeds on the way to the output, and the ${flow} shows the share of the relevance that comes back to it and where it goes from there, down to the pixels.`
          : `The ${flow} shows where its relevance goes from there, down to the pixels.`);
        if (v.boost > 1) parts.push(`Red line widths are scaled up ${v.boost < 10 ? v.boost.toFixed(1) : Math.round(v.boost)} times for this small share.`);
      } else {
        const st = startOf(cls), pn = cls !== pred ? ` (the network predicts ${pred})` : '';
        if (k !== null) parts.push(`<b>Class ${k}</b> has a score of <b>${net.logits[k].toFixed(1).replace('-', '−')}</b> and a probability of <b>${(probs[k] * 100).toFixed(probs[k] < 0.01 ? 2 : 0)}%</b>.`);
        if (isLRP() && st === 0) parts.push(`The score of class ${cls}${pn} is negative, and the z⁺ rule only follows contributions that raised it, so the relevance starts with <b>0.00</b> and there is nothing to pass back.`);
        else if (isLRP()) parts.push(`The relevance starts with <b>${fmtTot(st)}</b> in place of the score of class ${cls}${pn}, so that conservation is easy to see.${st < 0 ? ' The start is negative because the score is.' : ''} <b>${fmtTot(v.tot[0])}</b> of it reaches the pixels.`);
        else parts.push(`The gradient starts at <b>1.00</b> at the score of class ${cls}${pn}, because the score changes one for one with itself. <b>${fmtTot(v.tot[0])}</b> arrives at the pixels.`);
        parts.push({
          rules: st !== 0 && root.querySelector('.lrpw-sign') ? 'Every amount stays positive, because the z⁺ rule only follows positive contributions.' : '',
          lrp0: 'With LRP-0, an input that worked against the score receives an amount of the opposite sign. The relevance is still only divided, so the total stays the same from column to column.',
          gradient: 'Passed back like a gradient, the amount changes from layer to layer.',
          guided: 'Guided backpropagation passes back only the positive part of the gradient at every unit, and the amount still changes from layer to layer. Blue in a column is what arrives there before its units drop it. The pixels drop nothing, so some blue remains in the final map.',
        }[state.mode]);
        if (state.mode !== 'rules') parts.push('Blue lines and pixels carry negative amounts.');
      }
      readoutEl.innerHTML = parts.join(' ');
    };


    const load = () => {
      state.hover = null; state.classHover = null; fulls.clear(); viewSets.clear();
      net = forward(state.pixels);
      pred = cls = net.logits.indexOf(Math.max(...net.logits));
      const m = Math.max(...net.logits), ex = Array.from(net.logits, (v) => Math.exp(v - m)), total = sum(ex);
      probs = ex.map((v) => v / total);
      build();
      refresh();
      paint();
    };

    /* ---------- drawing on the input ---------- */
    // ink: a 280 x 280 buffer, white on black like MNIST, that fills the 20 x 20 digit box of the 28 x 28 input.
    // pad: a transparent canvas over the input square that shows the strokes while the pen is down.
    const pctx = pad.getContext('2d'), P = pad.width;
    const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const ink = canvas(P, P), ictx = ink.getContext('2d');
    const blank = () => { ictx.fillStyle = '#000'; ictx.fillRect(0, 0, P, P); };
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
    narrow.addEventListener('change', load);
  };

  const init = () => document.querySelectorAll('[data-lrp-water]').forEach((el) => {
    if (el.dataset.ready) return;
    el.dataset.ready = 'true';
    figure(el);
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

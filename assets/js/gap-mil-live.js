/* Live training figures of the MIL post: one label per picture, trained in the browser (no precomputed data).
   Toy SynMIL: pictures of 5 x 5 cells holding 3 to 5 different shapes out of ten; each picture is labelled with ONE of its
   shapes, picked at random. The model is the post's classifier in miniature: an encoder applied to every cell
   (12 x 12 pixels -> 32 features, ReLU), GAP over the 25 cells, a linear head to 10 scores (11 with background),
   cross-entropy on the one label, Adam. Nothing is precomputed: the data are drawn and the model trained in the page.
   data-live-mil="background" adds the background class and a control for how pictures are labelled. */
(() => {
  'use strict';
  if (window.GapMilLive) return;
  window.GapMilLive = true;
  const K = 10, G = 5, N = G * G, P = 12, D = P * P, H = 32;
  const NAMES = ['circle', 'ring', 'square', 'triangle', 'plus', 'star', 'diamond', 'cross', 'bar', 'T', 'background'];
  const BG = K;
  const EPOCHS = 20, BATCH = 16, NTRAIN = 3200, NTEST = 400, LR = 0.004, NOISE = 0.3;

  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    el.append(...kids.flat().filter((x) => x != null && x !== false));
    return el;
  };
  const svg = (tag, attrs = {}) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };
  const mulberry = (seed) => () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const shuffle = (a, rnd) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const range = (n) => [...Array(n).keys()];
  const pct = (x) => `${Math.round(100 * x)}%`;

  const shape = (ctx, cls) => {
    ctx.beginPath();
    const poly = (pts) => pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    const ngon = (n, r, a0 = -Math.PI / 2) => Array.from({ length: n }, (_, i) => [r * Math.cos(a0 + 2 * Math.PI * i / n), r * Math.sin(a0 + 2 * Math.PI * i / n)]);
    if (cls === 0) ctx.arc(0, 0, 1, 0, 2 * Math.PI);
    else if (cls === 1) { ctx.arc(0, 0, 1, 0, 2 * Math.PI); ctx.moveTo(0.52, 0); ctx.arc(0, 0, 0.52, 0, 2 * Math.PI, true); }
    else if (cls === 2) ctx.rect(-0.8, -0.8, 1.6, 1.6);
    else if (cls === 3) poly(ngon(3, 1.08, -Math.PI / 2).map(([x, y]) => [x, y + 0.15]));
    else if (cls === 4) poly([[-0.28, -1], [0.28, -1], [0.28, -0.28], [1, -0.28], [1, 0.28], [0.28, 0.28], [0.28, 1], [-0.28, 1], [-0.28, 0.28], [-1, 0.28], [-1, -0.28], [-0.28, -0.28]]);
    else if (cls === 5) poly(Array.from({ length: 10 }, (_, i) => { const r = i % 2 ? 0.45 : 1.08, a = -Math.PI / 2 + Math.PI * i / 5; return [r * Math.cos(a), r * Math.sin(a)]; }));
    else if (cls === 6) poly([[0, -1.05], [1.05, 0], [0, 1.05], [-1.05, 0]]);
    else if (cls === 7) poly([[-0.26, -1.05], [0.26, -1.05], [0.26, -0.26], [1.05, -0.26], [1.05, 0.26], [0.26, 0.26], [0.26, 1.05], [-0.26, 1.05], [-0.26, 0.26], [-1.05, 0.26], [-1.05, -0.26], [-0.26, -0.26]]
      .map(([x, y]) => [(x - y) / Math.SQRT2, (x + y) / Math.SQRT2]));
    else if (cls === 8) ctx.rect(-1, -0.34, 2, 0.68);
    else poly([[-1, -1], [1, -1], [1, -0.45], [0.27, -0.45], [0.27, 1], [-0.27, 1], [-0.27, -0.45], [-1, -0.45]]);
    ctx.closePath();
  };
  const place = (ctx, inst, cx, cy, unit) => {
    ctx.save();
    ctx.translate(cx + inst.dx * unit, cy + inst.dy * unit);
    ctx.rotate(inst.rot);
    ctx.scale(inst.size * unit, inst.size * unit);
    shape(ctx, inst.cls);
    ctx.restore();
  };

  const raster = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = P;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    return (inst) => {
      const x = new Float32Array(D);
      if (inst) {
        ctx.clearRect(0, 0, P, P);
        ctx.fillStyle = '#000';
        place(ctx, inst, P / 2, P / 2, P / 2);
        ctx.fill('evenodd');
        const px = ctx.getImageData(0, 0, P, P).data;
        for (let i = 0; i < D; i++) x[i] = px[4 * i + 3] / 255;
      }
      return x;
    };
  })();
  const makeBag = (rnd, mode = 'none') => {
    const empty = mode === 'empty' && rnd() < 0.1;
    const n = empty ? 0 : 3 + Math.floor(rnd() * 3);
    const classes = shuffle([...Array(K).keys()], rnd).slice(0, n);
    const cells = shuffle([...Array(N).keys()], rnd).slice(0, n);
    const items = Array(N).fill(null);
    classes.forEach((cls, j) => { items[cells[j]] = { cls, rot: (rnd() - 0.5) * 0.52, size: 0.62 + rnd() * 0.3, dx: (rnd() - 0.5) * 0.25, dy: (rnd() - 0.5) * 0.25 }; });
    let label = empty ? BG : classes[Math.floor(rnd() * n)];
    if (mode === 'label' && rnd() < 1 / (n + 1)) label = BG;
    return { items, classes, label, clean: items.map((it) => (it ? raster(it) : null)) };
  };
  const noisy = (bag, rnd) => bag.clean.map((c) => { const x = new Float32Array(D); for (let i = 0; i < D; i++) x[i] = (c ? c[i] : 0) + (rnd() - 0.5) * NOISE; return x; });

  const model = (rnd, K) => {
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const W1 = Float32Array.from({ length: H * D }, () => gauss() * Math.sqrt(2 / D)), b1 = new Float32Array(H);
    const W2 = Float32Array.from({ length: K * H }, () => gauss() * Math.sqrt(1 / H)), b2 = new Float32Array(K);
    const params = [W1, b1, W2, b2];
    const m = params.map((p) => new Float32Array(p.length)), v = params.map((p) => new Float32Array(p.length));
    let t = 0;
    const encode = (x, pre, hid) => {
      for (let j = 0; j < H; j++) {
        let s = b1[j];
        const o = j * D;
        for (let i = 0; i < D; i++) s += W1[o + i] * x[i];
        pre[j] = s; hid[j] = s > 0 ? s : 0;
      }
    };
    const head = (hid, out) => { for (let k = 0; k < K; k++) { let s = b2[k]; for (let j = 0; j < H; j++) s += W2[k * H + j] * hid[j]; out[k] = s; } };
    const readout = (bag) => {
      const pre = new Float32Array(H), hid = new Float32Array(H);
      const Z = bag.x.map((x) => { encode(x, pre, hid); const z = new Float32Array(K); head(hid, z); return z; });
      const z = new Float32Array(K);
      for (const zi of Z) for (let k = 0; k < K; k++) z[k] += zi[k] / N;
      return { Z, z };
    };
    const step = (bags, rnd) => {
      bags = bags.map((b) => ({ ...b, x: noisy(b, rnd) }));
      const g = params.map((p) => new Float32Array(p.length)), [gW1, gb1, gW2, gb2] = g;
      const pres = Array.from({ length: N }, () => new Float32Array(H)), hids = Array.from({ length: N }, () => new Float32Array(H));
      const hbar = new Float32Array(H), z = new Float32Array(K), dh = new Float32Array(H);
      let loss = 0;
      for (const bag of bags) {
        hbar.fill(0);
        for (let i = 0; i < N; i++) { encode(bag.x[i], pres[i], hids[i]); for (let j = 0; j < H; j++) hbar[j] += hids[i][j] / N; }
        head(hbar, z);
        const mx = Math.max(...z);
        let se = 0;
        for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); se += z[k]; }
        for (let k = 0; k < K; k++) z[k] /= se;
        loss -= Math.log(z[bag.label] + 1e-12);
        z[bag.label] -= 1;
        dh.fill(0);
        for (let k = 0; k < K; k++) { gb2[k] += z[k]; for (let j = 0; j < H; j++) { gW2[k * H + j] += z[k] * hbar[j]; dh[j] += W2[k * H + j] * z[k]; } }
        for (let i = 0; i < N; i++) {
          const x = bag.x[i], pre = pres[i];
          for (let j = 0; j < H; j++) {
            if (pre[j] <= 0) continue;
            const d = dh[j] / N, o = j * D;
            gb1[j] += d;
            for (let q = 0; q < D; q++) gW1[o + q] += d * x[q];
          }
        }
      }
      t++;
      const b1c = 1 - 0.9 ** t, b2c = 1 - 0.999 ** t;
      params.forEach((p, a) => {
        const ga = g[a], ma = m[a], va = v[a];
        for (let i = 0; i < p.length; i++) {
          const gi = ga[i] / bags.length;
          ma[i] = 0.9 * ma[i] + 0.1 * gi; va[i] = 0.999 * va[i] + 0.001 * gi * gi;
          p[i] -= LR * (ma[i] / b1c) / (Math.sqrt(va[i] / b2c) + 1e-8);
        }
      });
      return loss / bags.length;
    };
    return { readout, step };
  };
  const argmax = (a) => { let b = 0; for (let k = 1; k < a.length; k++) if (a[k] > a[b]) b = k; return b; };
  const softmax = (a) => { const mx = Math.max(...a), e = Array.from(a, (x) => Math.exp(x - mx)), s = e.reduce((p, q) => p + q, 0); return e.map((x) => x / s); };
  const evaluate = (net, bags) => {
    let img = 0, cells = 0, fg = 0, bg = 0, nbg = 0;
    for (const bag of bags) {
      const { Z, z } = net.readout(bag);
      img += argmax(z) === bag.label;
      bag.items.forEach((it, i) => { if (it) { fg++; cells += argmax(Z[i]) === it.cls; } else { nbg++; bg += argmax(Z[i]) === BG; } });
    }
    return { img: img / bags.length, cells: cells / fg, bg: bg / nbg };
  };

  const css = (name, fallback, el = document.documentElement) => getComputedStyle(el).getPropertyValue(name).trim() || fallback;
  const RIGHT = '#14857a', WRONG = '#d1495b';
  const where = (i) => `(${(i % G) + 1}, ${Math.floor(i / G) + 1})`;
  const fmt = (v) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`;
  const topk = (a, n) => Array.from(a, (v, k) => [k, v]).sort((p, q) => q[1] - p[1]).slice(0, n);
  const fit = (canvas) => {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth;
    if (canvas.width !== Math.round(W * dpr)) canvas.width = canvas.height = Math.round(W * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, W);
    return [ctx, W];
  };
  const drawThumb = (canvas, bag) => {
    const [ctx, W] = fit(canvas), c = W / G;
    ctx.fillStyle = css('--lm-img-bg', '#fff', canvas);
    ctx.fillRect(0, 0, W, W);
    ctx.fillStyle = css('--lm-img-ink', '#24211d', canvas);
    bag.items.forEach((it, i) => { if (it) { place(ctx, it, (i % G + 0.5) * c, (Math.floor(i / G) + 0.5) * c, c * 0.36); ctx.fill('evenodd'); } });
  };
  const drawReadout = (canvas, bag, ro, hover, bgClass = false) => {
    const [ctx, W] = fit(canvas), c = W / G, gap = Math.max(2, W / 160), ink = css('--lm-img-ink', '#24211d', canvas), muted = css('--lm-img-muted', '#8b867c', canvas), bg = css('--lm-img-bg', '#fff', canvas);
    const font = getComputedStyle(canvas).fontFamily;
    const tile = (x, y) => { ctx.beginPath(); ctx.rect(x + gap / 2, y + gap / 2, c - gap, c - gap); };
    ctx.textAlign = 'center';
    for (let i = 0; i < N; i++) {
      const x = (i % G) * c, y = Math.floor(i / G) * c, it = bag.items[i];
      const p = softmax(ro.Z[i]), k = argmax(p), conf = Math.max(0, (p[k] - 1 / p.length) / (1 - 1 / p.length));
      ctx.fillStyle = bg;
      tile(x, y); ctx.fill();
      if (it) {
        ctx.fillStyle = k === it.cls ? RIGHT : WRONG;
        ctx.globalAlpha = 0.18 + 0.55 * conf;
        tile(x, y); ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink;
        place(ctx, it, x + c / 2, y + c * 0.43, c * 0.27); ctx.fill('evenodd');
        ctx.font = `600 ${Math.round(c * 0.15)}px ${font}`;
        ctx.fillText(NAMES[k], x + c / 2, y + c * 0.88);
      } else if (bgClass) {
        ctx.fillStyle = k === BG ? RIGHT : WRONG;
        ctx.globalAlpha = 0.12 + 0.45 * conf;
        tile(x, y); ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink;
        ctx.font = `500 ${Math.round(c * 0.13)}px ${font}`;
        ctx.fillText(NAMES[k], x + c / 2, y + c * 0.56);
      } else {
        ctx.fillStyle = muted;
        ctx.globalAlpha = 0.7;
        ctx.font = `500 ${Math.round(c * 0.14)}px ${font}`;
        ctx.fillText(NAMES[k], x + c / 2, y + c * 0.56);
        ctx.globalAlpha = 1;
      }
    }
    if (hover != null) {
      const x = (hover % G) * c, y = Math.floor(hover / G) * c, lw = 2.5;
      ctx.strokeStyle = ink; ctx.lineWidth = lw;
      ctx.strokeRect(x + gap / 2 + lw / 2, y + gap / 2 + lw / 2, c - gap - lw, c - gap - lw);
    }
  };

  const trainer = (mode = 'none') => {
    const T = { mode, steps: 0, running: false, perEpoch: NTRAIN / BATCH, total: EPOCHS * NTRAIN / BATCH, fns: [] };
    let rnd, raf = 0;
    T.on = (fn) => T.fns.push(fn);
    T.emit = () => { if (T.train) T.fns.forEach((fn) => fn()); };
    T.init = () => {
      rnd = mulberry(20261126);
      T.bgClass = T.mode !== 'none';
      T.train = Array.from({ length: NTRAIN }, () => makeBag(rnd, T.mode));
      const trnd = mulberry(4242);
      T.test = Array.from({ length: NTEST }, () => makeBag(trnd)).map((b) => ({ ...b, x: noisy(b, trnd) }));
      T.net = model(mulberry(7), T.bgClass ? K + 1 : K);
      T.steps = 0; T.strip = T.train.slice(0, BATCH);
      T.history = [{ e: 0, ...evaluate(T.net, T.test) }];
      T.ceiling = T.test.reduce((s, b) => s + 1 / b.classes.length, 0) / T.test.length;
      T.emit();
    };
    T.smooth = () => {
      const out = [];
      for (const p of T.history) { const q = out[out.length - 1]; out.push(q ? { e: p.e, cells: 0.6 * q.cells + 0.4 * p.cells, img: 0.6 * q.img + 0.4 * p.img, bg: 0.6 * q.bg + 0.4 * p.bg } : { ...p }); }
      return out;
    };
    const tick = () => {
      if (!T.running) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 12 && T.steps < T.total) {
        const order = (T.steps % T.perEpoch) * BATCH;
        if (order === 0) shuffle(T.train, rnd);
        T.net.step(T.train.slice(order, order + BATCH), rnd);
        T.steps++;
        if (T.steps % (T.perEpoch / 4) === 0) {
          T.history.push({ e: T.steps / T.perEpoch, ...evaluate(T.net, T.test) });
          T.strip = T.train.slice(order, order + BATCH);
          T.emit();
        }
      }
      if (T.steps >= T.total) { T.running = false; T.emit(); return; }
      raf = requestAnimationFrame(tick);
    };
    T.start = () => { if (T.steps >= T.total) T.init(); T.running = true; tick(); T.emit(); };
    T.stop = () => { T.running = false; cancelAnimationFrame(raf); T.emit(); };
    return T;
  };

  const controls = (T) => {
    const play = h('button', { type: 'button', class: 'lm-play', onclick: () => (T.running ? T.stop() : T.start()) });
    const bar = h('span', { class: 'lm-progress-fill' }), epoch = h('span', { class: 'lm-epoch' });
    const el = h('div', { class: 'lm-bar' }, play, h('span', { class: 'lm-progress' }, bar), epoch,
      h('button', { type: 'button', class: 'lm-reset', onclick: () => { T.stop(); T.init(); } }, 'Reset'));
    T.on(() => {
      play.textContent = T.running ? 'Pause' : T.steps >= T.total ? 'Train again' : T.steps ? 'Resume' : 'Train';
      play.classList.toggle('is-running', T.running);
      bar.style.width = `${100 * T.steps / T.total}%`;
      epoch.textContent = `epoch ${Math.floor(T.steps / T.perEpoch)}/${EPOCHS}`;
    });
    return el;
  };
  const gallery = (T, n) => {
    const shots = range(n).map(() => {
      const c = h('canvas', { class: 'lm-thumb', role: 'img' }), lab = h('span', { class: 'lm-tag' });
      return { c, lab, el: h('figure', { class: 'lm-shot' }, c, lab) };
    });
    T.on(() => shots.forEach((t, j) => {
      const bag = T.strip[j % T.strip.length];
      t.lab.textContent = NAMES[bag.label];
      t.c.setAttribute('aria-label', `A training picture labelled ${NAMES[bag.label]}`);
      drawThumb(t.c, bag);
    }));
    return h('div', {}, h('p', { class: 'lm-sub', text: `A training batch: ${n} of its ${BATCH} pictures` }), h('div', { class: 'lm-gallery' }, shots.map((t) => t.el)));
  };
  const stats = (T) => {
    const bg = T.mode !== 'none';
    const a = h('strong', { class: 'lm-big lm-right' }), b = h('strong', { class: 'lm-big' }), what = h('span');
    T.on(() => {
      const sm = T.smooth(), last = sm[sm.length - 1];
      a.textContent = pct(last.cells); b.textContent = pct(bg ? last.bg : last.img);
      what.textContent = bg ? 'of empty cells right' : `images right (max ${pct(T.ceiling)})`;
    });
    return h('div', { class: 'lm-stats' },
      h('div', { class: 'lm-stat' }, a, h('span', { text: 'of shapes named right' })),
      h('div', { class: `lm-stat${bg ? ' lm-stat-bg' : ''}` }, b, what));
  };
  const chart = (T) => {
    const el = svg('svg', { class: 'lm-chart', role: 'img', 'aria-label': 'Accuracy during training' });
    const box = h('div', { class: 'lm-chartbox' }, el);
    const draw = () => {
      if (!T.history) return;
      const r = box.getBoundingClientRect(), W = Math.max(200, Math.round(r.width)), H2 = Math.max(120, Math.round(r.height));
      el.setAttribute('viewBox', `0 0 ${W} ${H2}`);
      const L = 34, R = 10, Tp = 10, B = 24, x = (e) => L + (W - L - R) * e / EPOCHS, y = (v) => Tp + (H2 - Tp - B) * (1 - v);
      el.replaceChildren();
      for (const v of [0, 0.5, 1]) {
        el.append(svg('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'lm-gridline' }));
        const t = svg('text', { x: L - 6, y: y(v) + 3.5, class: 'lm-axis', 'text-anchor': 'end' }); t.textContent = pct(v); el.append(t);
      }
      for (const e of [0, 5, 10, 15, 20]) { const t = svg('text', { x: x(e), y: H2 - 6, class: 'lm-axis', 'text-anchor': 'middle' }); t.textContent = e ? e : 'epoch 0'; el.append(t); }
      const second = T.mode !== 'none' ? 'bg' : 'img';
      if (second === 'img') {
        el.append(svg('line', { x1: L, x2: W - R, y1: y(T.ceiling), y2: y(T.ceiling), class: 'lm-ceiling' }));
        const ct = svg('text', { x: L + 6, y: y(T.ceiling) - 6, class: 'lm-axis' }); ct.textContent = 'best possible from one label'; el.append(ct);
      }
      const sm = T.smooth(), last = sm[sm.length - 1], path = (k) => sm.map((p, i) => `${i ? 'L' : 'M'}${x(p.e).toFixed(1)},${y(p[k]).toFixed(1)}`).join('');
      el.append(svg('path', { d: `${path('cells')}L${x(last.e).toFixed(1)},${y(0)}L${x(0)},${y(0)}Z`, class: 'lm-area' }));
      const cls2 = second === 'bg' ? 'lm-l-bg' : 'lm-l-img';
      el.append(svg('path', { d: path(second), class: `lm-line ${cls2}` }), svg('path', { d: path('cells'), class: 'lm-line lm-l-cells' }));
      for (const [k, cls] of [[second, cls2], ['cells', 'lm-l-cells']]) el.append(svg('circle', { cx: x(last.e), cy: y(last[k]), r: 3, class: `lm-dot ${cls}` }));
    };
    T.on(draw);
    new ResizeObserver(draw).observe(box);
    return box;
  };
  const predict = (T) => {
    let shown = 0, hover = null;
    const canvas = h('canvas', { class: 'lm-test', role: 'img', 'aria-label': 'A test picture: each shape cell names its shape' });
    const another = h('button', { type: 'button', class: 'lm-reset', onclick: () => { shown = (shown + 1) % T.test.length; hover = null; paint(); } }, 'Another test picture');
    const imgList = h('ol', { class: 'gm-list lm-list' }), cellHead = h('p', { class: 'gm-h' }), cellBody = h('div');
    const rows = (pairs, right) => {
      const max = Math.max(1e-9, ...pairs.map(([, v]) => v));
      return pairs.map(([k, v]) => h('li', {}, h('div', { class: `gm-row lm-row${right(k) ? '' : ' is-dim'}` },
        h('span', { class: 'gm-name', text: NAMES[k] }), h('span', { class: 'gm-val', text: fmt(v) }),
        h('span', { class: 'gm-track' }, h('span', { class: 'gm-fill', style: `width:${Math.max(0, v / max * 100)}%;background:${right(k) ? RIGHT : 'var(--lm-grey)'}` })))));
    };
    const cellAt = (e) => { const r = canvas.getBoundingClientRect(), c = r.width / G, col = Math.floor((e.clientX - r.left) / c), row = Math.floor((e.clientY - r.top) / c); return col >= 0 && col < G && row >= 0 && row < G ? row * G + col : null; };
    canvas.addEventListener('pointermove', (e) => { if (e.pointerType !== 'mouse') return; const i = cellAt(e); if (i !== hover) { hover = i; paint(); } });
    canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse' && hover != null) { hover = null; paint(); } });
    canvas.addEventListener('pointerdown', (e) => { if (e.pointerType === 'mouse') return; const i = cellAt(e); hover = i === hover ? null : i; paint(); });
    function paint() {
      if (!T.test) return;
      const bag = T.test[shown], ro = T.net.readout(bag);
      drawReadout(canvas, bag, ro, hover, T.bgClass);
      const present = (k) => bag.classes.includes(k) || (T.bgClass && k === BG);
      imgList.replaceChildren(...rows(topk(ro.z, 6), present));
      if (hover == null) {
        cellHead.textContent = 'A cell';
        cellBody.replaceChildren(h('p', { class: 'lm-hint', text: 'Hover or tap a cell to see its six highest scores.' }));
      } else {
        const it = bag.items[hover], truth = it ? it.cls : T.bgClass ? BG : null;
        cellHead.textContent = `Cell at ${where(hover)}`;
        cellBody.replaceChildren(h('ol', { class: 'gm-list lm-list' }, rows(topk(ro.Z[hover], 6), (k) => k === truth)));
      }
    }
    T.on(paint);
    new ResizeObserver(paint).observe(canvas);
    return {
      picture: canvas,
      another,
      image: h('div', { class: 'lm-card' }, h('p', { class: 'gm-h', text: 'Image' }), imgList),
      cell: h('div', { class: 'lm-card' }, cellHead, cellBody),
    };
  };
  const kicker = (num, title, extra) => h('p', { class: 'lm-kicker' }, h('span', { class: 'lm-num', text: num }), h('b', { text: title }), extra ? h('span', { class: 'lm-kicker-extra' }, extra) : null);

  const MODES = [['never', 'Class only'], ['label', 'As a label'], ['empty', 'Empty pictures']];
  const modeHelp = {
    never: 'A background class, but no picture is labelled with it.',
    label: 'Background is in every picture, so it is sometimes the label.',
    empty: 'One picture in ten is empty and labelled background.',
  };
  const modes = (T) => {
    const btns = MODES.map(([m, t]) => h('button', { type: 'button', 'aria-pressed': String(T.mode === m), onclick: () => { T.stop(); T.mode = m; T.init(); } }, t));
    const help = h('p', { class: 'lm-sub' });
    T.on(() => { btns.forEach((b, j) => b.setAttribute('aria-pressed', String(MODES[j][0] === T.mode))); help.textContent = modeHelp[T.mode]; });
    return h('div', { class: 'lm-modes' }, h('div', { class: 'lm-seg', role: 'group', 'aria-label': 'Training labels' }, btns), help);
  };
  const layout = (app, T) => {
    const pr = predict(T);
    app.append(h('div', { class: 'lm-c' },
      h('section', { class: 'lm-c-card' }, kicker('1', 'Training'), T.mode !== 'none' ? modes(T) : null, controls(T), gallery(T, 8), stats(T), chart(T)),
      h('section', { class: 'lm-c-card' }, kicker('2', 'Prediction', pr.another), pr.picture, h('div', { class: 'lm-c-lists' }, pr.image, pr.cell))));
  };

  document.querySelectorAll('[data-live-mil]').forEach((app) => {
    const T = trainer(app.dataset.liveMil === 'background' ? 'empty' : 'none');
    app.replaceChildren();
    layout(app, T);
    T.init();
    new MutationObserver(() => T.emit()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  });
})();

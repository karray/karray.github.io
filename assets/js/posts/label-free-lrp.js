/* Interactive figures for the LRP post. The stacked layers read precomputed relevance; the toys compute in the page. */
(() => {
  'use strict';

  const h = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    children.flat(Infinity).forEach((c) => c !== null && c !== undefined && c !== false && node.append(c));
    return node;
  };
  const svg = (tag, attrs = {}) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
  const fmt = (v, digits = 2) => {
    if (!Number.isFinite(v)) return '∞';
    const r = Number(v.toFixed(digits));
    return (Object.is(r, -0) ? 0 : r).toFixed(digits).replace('-', '−');
  };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* matplotlib's "hot" colormap */
  const hot = (t) => [clamp(t / 0.365, 0, 1), clamp((t - 0.365) / 0.381, 0, 1), clamp((t - 0.746) / 0.254, 0, 1)];

  /* A horizontal bar for a signed amount. Positive grows to the right of the zero line, negative to the left. */
  const bar = (max = 1.5, { signed = true } = {}) => {
    const fill = h('span', { class: 'lrp-bar-fill' });
    const value = h('output', { class: 'lrp-bar-value' });
    const track = h('span', { class: `lrp-bar-track${signed ? ' lrp-bar-track--signed' : ''}` }, fill);
    const root = h('span', { class: 'lrp-bar' }, track, value);
    return {
      root,
      set(v, text) {
        const frac = Number.isFinite(v) ? Math.min(Math.abs(v) / max, 1) : 1;
        const half = signed ? 50 : 100;
        fill.style.width = `${frac * half}%`;
        fill.style.left = signed ? (v < 0 ? `${50 - frac * 50}%` : '50%') : '0';
        root.classList.toggle('lrp-neg', v < 0);
        root.classList.toggle('lrp-over', !Number.isFinite(v) || Math.abs(v) > max);
        value.textContent = text ?? fmt(v);
      },
    };
  };

  const slider = (label, { min, max, step = 0.01, value, onInput, digits = 2 }) => {
    const input = h('input', { type: 'range', min, max, step, value, 'aria-label': label });
    const out = h('output', { text: fmt(Number(value), digits) });
    input.addEventListener('input', () => {
      out.textContent = fmt(Number(input.value), digits);
      onInput(Number(input.value));
    });
    const root = h('label', { class: 'lrp-slider' }, h('span', { class: 'lrp-slider-label', text: label }), input, out);
    return {
      root, input,
      set(v) { input.value = v; out.textContent = fmt(Number(input.value), digits); },
      get value() { return Number(input.value); },
    };
  };

  /* ------------------------------------------------------------------------------------- math */
  const erf = (x) => {      // Abramowitz and Stegun 7.1.26
    const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  };
  const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const phi = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  const gelu = (x) => x * Phi(x);
  const geluSlope = (x) => Phi(x) + x * phi(x);
  const stab = (z, e = 1e-6) => z + (z >= 0 ? e : -e);

  /* ---------------------------------------------------------------------------- activation toy */
  const activationToy = (root) => {
    const W = 320, H = 200, X0 = -3, X1 = 3, Y0 = -0.6, Y1 = 3;
    const sx = (x) => 30 + (x - X0) / (X1 - X0) * (W - 40);
    const sy = (y) => H - 24 - (y - Y0) / (Y1 - Y0) * (H - 34);
    const plot = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'The GELU curve with the chosen input marked' });
    plot.append(svg('line', { x1: sx(X0), x2: sx(X1), y1: sy(0), y2: sy(0), class: 'lrp-axis' }));
    plot.append(svg('line', { x1: sx(0), x2: sx(0), y1: sy(Y0), y2: sy(Y1), class: 'lrp-axis' }));
    let d = '';
    for (let i = 0; i <= 120; i++) {
      const x = X0 + (X1 - X0) * i / 120;
      d += `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(gelu(x)).toFixed(1)}`;
    }
    plot.append(svg('path', { d, class: 'lrp-curve' }));
    const tangent = svg('line', { class: 'lrp-tangent' });
    const dot = svg('circle', { r: 6, class: 'lrp-dot' });
    plot.append(tangent, dot);
    const lbl = (x, y, t, a = 'start') => { const n = svg('text', { x, y, class: 'lrp-svg-label', 'text-anchor': a }); n.textContent = t; return n; };
    plot.append(lbl(W - 8, sy(0) - 6, 'input', 'end'), lbl(sx(0) + 6, 14, 'output'));

    const keep = bar(1.2, { signed: true });
    const pass = bar(1.2, { signed: true });
    const note = h('p', { class: 'lrp-result', 'aria-live': 'polite' });
    const update = (x) => {
      const y = gelu(x), k = geluSlope(x), dx = 0.9;
      dot.setAttribute('cx', sx(x)); dot.setAttribute('cy', sy(y));
      tangent.setAttribute('x1', sx(x - dx)); tangent.setAttribute('y1', sy(y - k * dx));
      tangent.setAttribute('x2', sx(x + dx)); tangent.setAttribute('y2', sy(y + k * dx));
      keep.set(1);
      pass.set(k);
      const lost = 1 - k;
      note.textContent = lost > 0.005
        ? `At this input the curve has a slope of ${fmt(k)}. Without a rule, ${Math.round(lost * 100)}% of the relevance is lost at this one value.`
        : lost < -0.005
          ? `At this input the curve is steeper than 1 (slope ${fmt(k)}), so without a rule the relevance grows by ${Math.round(-lost * 100)}%.`
          : 'Here the slope is close to 1, so both treatments agree.';
    };
    const s = slider('Input value', { min: -3, max: 3, value: -0.5, onInput: update });
    root.append(h('div', { class: 'lrp-columns' },
      h('div', {}, plot, s.root),
      h('div', { class: 'lrp-readouts' },
        h('p', { class: 'lrp-row-title', text: 'Relevance arriving at the output' }), (() => { const b = bar(1.2); b.set(1); return b.root; })(),
        h('p', { class: 'lrp-row-title', text: 'Passed back with the identity rule' }), keep.root,
        h('p', { class: 'lrp-row-title', text: 'Passed back without a rule (times the slope)' }), pass.root,
        note)));
    update(s.value);
  };

  /* ----------------------------------------------------------------------------- LayerNorm toy */
  const layerNormToy = (root) => {
    const start = [1.6, 0.3, -0.4, 2.2, -1.0, 0.7];
    let x = start.slice();
    const n = x.length;
    const cols = ['Output', 'With the rule', 'Without a rule'];
    const outBars = x.map(() => bar(2.5)), ruleBars = x.map(() => bar(1.5)), gradBars = x.map(() => bar(1.5));
    const totals = cols.map(() => h('output', { class: 'lrp-total' }));
    const sliders = x.map((v, i) => slider(`Input ${i + 1}`, { min: -2, max: 4, value: v, onInput: (val) => { x[i] = val; update(); } }));
    const note = h('p', { class: 'lrp-result', 'aria-live': 'polite' });

    const update = () => {
      const mu = x.reduce((a, b) => a + b, 0) / n;
      const sigma = Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / n + 1e-5);
      const y = x.map((v) => (v - mu) / sigma);
      const pos = y.map((v) => Math.max(v, 0)), S = pos.reduce((a, b) => a + b, 0) || 1;
      const R = pos.map((v) => v / S);                                  // relevance arriving at the outputs
      const ratio = R.map((r, i) => r / stab(y[i]));
      const ratioSum = ratio.reduce((a, b) => a + b, 0);
      const rule = x.map((xj, j) => xj / sigma * (ratio[j] - ratioSum / n));       // spread held fixed
      const Rsum = R.reduce((a, b) => a + b, 0), Ry = R.reduce((a, r, i) => a + r * y[i], 0);
      const grad = y.map((yj, j) => (R[j] - Rsum / n - yj * Ry / n) / sigma);       // the gradient of LayerNorm
      y.forEach((v, i) => { outBars[i].set(v); ruleBars[i].set(rule[i]); gradBars[i].set(grad[i]); });
      const sum = (a) => a.reduce((p, q) => p + q, 0);
      totals[0].textContent = `relevance arriving ${fmt(sum(R))}`;
      totals[1].textContent = `total ${fmt(sum(rule))}`;
      totals[2].textContent = `total ${fmt(sum(grad))}`;
      note.textContent = `Without a rule, the inputs receive ${fmt(sum(grad.filter((g) => g > 0)))} of positive and ${fmt(sum(grad.filter((g) => g < 0)))} of negative relevance, which cancel exactly. With the spread held fixed, the total of ${fmt(sum(rule))} is kept.`;
    };
    const shift = () => { x = x.map((v) => clamp(v + 1, -2, 4)); sliders.forEach((s, i) => s.set(x[i])); update(); };
    const scale = () => { x = x.map((v) => clamp(v * 1.5, -2, 4)); sliders.forEach((s, i) => s.set(x[i])); update(); };
    const reset = () => { x = start.slice(); sliders.forEach((s, i) => s.set(x[i])); update(); };

    const grid = h('div', { class: 'lrp-table', style: '--lrp-cols: 4' },
      h('span', { class: 'lrp-head', text: 'Input' }), cols.map((c) => h('span', { class: 'lrp-head', text: c })),
      x.map((_, i) => [sliders[i].root, outBars[i].root, ruleBars[i].root, gradBars[i].root]),
      h('span'), totals);
    root.append(grid,
      h('div', { class: 'lrp-buttons' },
        h('button', { type: 'button', onclick: shift, text: 'Add 1 to every input' }),
        h('button', { type: 'button', onclick: scale, text: 'Stretch the inputs by 1.5' }),
        h('button', { type: 'button', onclick: reset, text: 'Reset' })),
      note);
    update();
  };

  /* ----------------------------------------------------------------------------- attention toy */
  const attentionToy = (root) => {
    const names = ['A', 'B', 'C', 'D'];
    const v = [1.5, 0.05, 0.6, 1.0], s = [0.3, 1.6, 0.6, -0.4];
    const wBars = v.map(() => bar(1, { signed: false })), sentBars = v.map(() => bar(1));
    const ruleBars = v.map(() => bar(1)), gradBars = v.map(() => bar(1));
    const totals = [0, 1, 2, 3].map(() => h('output', { class: 'lrp-total' }));
    const out = h('output');
    const note = h('p', { class: 'lrp-result', 'aria-live': 'polite' });
    const update = () => {
      const m = Math.max(...s), e = s.map((q) => Math.exp(q - m)), Z = e.reduce((a, b) => a + b, 0);
      const A = e.map((q) => q / Z), sent = A.map((a, j) => a * v[j]), o = sent.reduce((a, b) => a + b, 0);
      const rule = sent.map((c) => c / stab(o, 1e-3));
      const toScores = A.map((a, j) => a * (v[j] - o));
      A.forEach((a, j) => { wBars[j].set(a); sentBars[j].set(sent[j]); ruleBars[j].set(rule[j]); gradBars[j].set(a); });
      out.textContent = fmt(o);
      const sum = (q) => q.reduce((a, b) => a + b, 0);
      totals[0].textContent = `sum ${fmt(sum(A))}`;
      totals[1].textContent = `output ${fmt(o)}`;
      totals[2].textContent = `total ${fmt(sum(rule))}`;
      totals[3].textContent = `total ${fmt(sum(A))}, plus ${fmt(sum(toScores.filter((q) => q > 0)))} and ${fmt(sum(toScores.filter((q) => q < 0)))} into the weights`;
      const top = A.indexOf(Math.max(...A));
      note.textContent = `Token ${names[top]} has the largest weight (${fmt(A[top])}) and sends ${fmt(sent[top])}. With the weights held fixed it receives ${fmt(rule[top])} of the relevance, and without a rule ${fmt(A[top])}.`;
    };
    const rows = names.map((name, j) => [
      h('span', { class: 'lrp-token' }, h('strong', { text: `Token ${name}` }),
        slider('value', { min: -1, max: 2, value: v[j], onInput: (q) => { v[j] = q; update(); } }).root,
        slider('score', { min: -2, max: 2, value: s[j], onInput: (q) => { s[j] = q; update(); } }).root),
      wBars[j].root, sentBars[j].root, ruleBars[j].root, gradBars[j].root]);
    root.append(
      h('div', { class: 'lrp-table lrp-table--wide', style: '--lrp-cols: 5' },
        ['Source token', 'Attention weight', 'What it sends', 'Weights held fixed', 'Without a rule'].map((c) => h('span', { class: 'lrp-head', text: c })),
        rows, h('span'), totals),
      h('p', { class: 'lrp-small' }, 'The output token receives ', out, ' and holds a relevance of 1.00.'),
      note);
    update();
  };

  /* ------------------------------------------------------------------------------ residual toy */
  const residualToy = (root) => {
    let skip = 1.0, upd = 0.4;
    const all = {
      copy: ['Plain addition', () => [1, 1]],
      equal: ['Equal', () => [0.5, 0.5]],
      proportional: ['Proportional', (a, b) => { const y = stab(a + b); return [a / y, b / y]; }],
      magnitude: ['Magnitude', (a, b) => { const t = Math.abs(a) + Math.abs(b) || 1; return [Math.abs(a) / t, Math.abs(b) / t]; }],
      gamma: ['γ (ResLRP)', (a, b) => {
        const y = a + b, w = [a, b].map((z) => z * (1 + (Math.sign(z) === Math.sign(y) ? 1 : 0)));
        const d = stab(w[0] + w[1]); return [w[0] / d, w[1] / d];
      }],
    };
    const rules = (root.dataset.rules || 'copy,equal,proportional,magnitude,gamma').split(',').map((k) => all[k.trim()]).filter(Boolean);
    const presets = root.dataset.presets !== 'false';
    const bars = rules.map(() => [bar(2.5), bar(2.5)]);
    const totals = rules.map(() => h('output', { class: 'lrp-total' }));
    const out = h('output');
    const sSkip = slider('Skip path', { min: -2, max: 2, value: skip, onInput: (q) => { skip = q; update(); } });
    const sUpd = slider('Update', { min: -2, max: 2, value: upd, onInput: (q) => { upd = q; update(); } });
    const update = () => {
      out.textContent = fmt(skip + upd);
      rules.forEach(([, f], i) => {
        const [a, b] = f(skip, upd);
        bars[i][0].set(a); bars[i][1].set(b);
        totals[i].textContent = fmt(a + b);
      });
    };
    const preset = (a, b) => () => { skip = a; upd = b; sSkip.set(a); sUpd.set(b); update(); };
    root.append(h('div', {},
      h('div', { class: 'lrp-sliders' }, sSkip.root, sUpd.root),
      h('p', { class: 'lrp-small' }, 'The block outputs their sum, ', out, ', and holds a relevance of 1.00.'),
      presets ? h('div', { class: 'lrp-buttons' },
        h('button', { type: 'button', onclick: preset(1, 0.4), text: 'A small update' }),
        h('button', { type: 'button', onclick: preset(1, -0.9), text: 'The update almost cancels the input' }),
        h('button', { type: 'button', onclick: preset(-0.3, 1.2), text: 'The update outweighs the input' })) : null,
      h('div', { class: 'lrp-table', style: '--lrp-cols: 4' },
        ['Rule', 'To the skip path', 'To the update', 'Total'].map((c) => h('span', { class: 'lrp-head', text: c })),
        rules.map(([name], i) => [h('span', { class: 'lrp-rule', text: name }), bars[i][0].root, bars[i][1].root, totals[i]]))));
    totals.forEach((t) => t.classList.add('lrp-total--row'));
    update();
  };

  /* ------------------------------------------------------------------------- stacked layers */
  /* [where, short label] for a level name from the exported data */
  const LEVEL = (name) => {
    if (name === 'features') return ['the start', 'start'];
    if (name === 'pixels') return ['the pixels', 'pixels'];
    if (name === 'below the final norm') return ['below the final norm', 'final norm'];
    let m = name.match(/^below block (\d+)$/);
    if (m) return [`below block ${m[1]}`, `block ${m[1]}`];
    m = name.match(/→ (?:stage )?(\d+)$/);
    if (m) return [`below stage ${m[1]}`, `stage ${m[1]}`];
    return [name, name];
  };
  const SHOWN = ['below the final norm', 'below block 9', 'below block 5', 'below block 1', 'pixels'];
  const M = 112;                                    // display resolution of the layer maps
  const TILT = 30 * Math.PI / 180, DEPTH = 760;     // keep in sync with .lrp-plane in the CSS

  const stackExplorer = (root, app) => {
    const base = app.dataset.base;
    const models = JSON.parse(app.dataset.models);
    const thumbs = [...root.querySelectorAll('.fg-thumb')];
    const state = { model: models[0].id, item: thumbs[0]?.dataset.item, data: null, img: null, hover: -1, picked: null, common: false };
    const cache = new Map(), images = new Map();

    const modelSelect = h('select', { 'aria-label': 'Model' }, models.map((m) => h('option', { value: m.id, text: m.label })));
    const icon = (label, d, fill) => {
      const b = h('button', { type: 'button', class: 'lrp-icon', title: label, 'aria-label': label });
      const g = svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' });
      d.forEach((path, i) => g.append(svg('path', { d: path, class: fill && fill[i] ? 'lrp-icon-fill' : '' })));
      b.append(g);
      return b;
    };
    const invertBtn = icon('Invert the selection', ['M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16z', 'M12 4a8 8 0 0 1 0 16z'], [0, 1]);
    const resetBtn = icon('Reset: all cells, default scale and opacity', ['M4 12a8 8 0 1 0 2.4-5.7', 'M4 4v4.5h4.5']);
    const commonBox = h('input', { type: 'checkbox' });
    const opacity = h('input', { type: 'range', min: 0, max: 100, step: 5, value: 100, 'aria-label': 'Map opacity' });
    const opacityOut = h('output', { text: '100%' });
    const layer = (cls, size) => h('canvas', { class: `lrp-canvas ${cls}`, width: size, height: size });
    const startBase = layer('', 448), startHeat = layer('lrp-heat', 448), startMarks = layer('', 448);
    const startBox = h('div', { class: 'lrp-start', role: 'img', 'aria-label': 'The start. Hover a cell to preview it, click cells to select them.' }, startBase, startHeat, startMarks);
    const lanes = h('div', { class: 'lrp-layers' });
    const ribbons = svg('svg', { class: 'lrp-ribbons', 'aria-hidden': 'true' });
    const readout = h('p', { class: 'lrp-result', 'aria-live': 'polite' });
    const status = h('p', { class: 'lrp-small lrp-status' });
    app.append(
      h('div', { class: 'lrp-stack-controls' },
        h('label', { class: 'lrp-select' }, 'Model', modelSelect),
        h('label', { class: 'lrp-range' }, 'Map opacity', opacity, opacityOut),
        h('label', { class: 'lrp-check' }, commonBox, ' One colour scale for all depths')),
      h('div', { class: 'lrp-stage' },
        h('div', { class: 'lrp-startcol' },
          h('p', { class: 'lrp-layer-top', text: 'where it starts' }), startBox,
          h('div', { class: 'lrp-tools', role: 'toolbar', 'aria-label': 'Start cells' }, invertBtn, resetBtn),
          h('p', { class: 'lrp-layer-name', text: 'the features' })),
        h('div', { class: 'lrp-layerwrap' }, ribbons, lanes)),
      readout, status);

    const load = (model, item) => {
      const key = `${model}_${item}`;
      if (!cache.has(key)) {
        cache.set(key, (async () => {
          const [meta, buf] = await Promise.all([
            fetch(`${base}${key}.json`).then((r) => r.json()),
            fetch(`${base}${key}.dat`).then((r) => new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer()),
          ]);
          const q = new Int8Array(buf), cells = meta.grid[0] * meta.grid[1];
          let offset = 0;
          meta.levels.forEach((lv) => { const n = lv.size[0] * lv.size[1]; lv.q = q.subarray(offset, offset + cells * n); offset += cells * n; });
          meta.cells = cells;
          return meta;
        })());
      }
      return cache.get(key);
    };
    const image = (item) => {
      if (!images.has(item)) {
        images.set(item, new Promise((resolve) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.src = thumbs.find((t) => t.dataset.item === item).dataset.src;
        }));
      }
      return images.get(item);
    };

    const sumLevel = (lv, sel) => {
      const n = lv.size[0] * lv.size[1], out = new Float32Array(n);
      sel.forEach((on, c) => {
        if (!on) return;
        const s = lv.scale[c], o = c * n;
        for (let i = 0; i < n; i++) out[i] += lv.q[o + i] * s;
      });
      return out;
    };
    const amount = (lv, sel) => sel.reduce((a, on, c) => (on ? a + lv.net[c] + lv.cls[c] : a), 0);
    const toGrid = (v, [r, c], [gr, gc]) => {
      if (r === gr && c === gc) return v;
      const out = new Float32Array(gr * gc), fr = r / gr, fc = c / gc;
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[((i / fr) | 0) * gc + ((j / fc) | 0)] += v[i * c + j];
      return out;
    };
    const resample = (v, [r, c]) => {               // nearest, so token and stage grids stay visible
      const out = new Float32Array(M * M);
      for (let y = 0; y < M; y++) {
        const row = Math.min(r - 1, (y * r / M) | 0) * c;
        for (let x = 0; x < M; x++) out[y * M + x] = v[row + Math.min(c - 1, (x * c / M) | 0)];
      }
      return out;
    };
    const selection = () => {                       // clicked cells plus the hovered one; nothing clicked = all
      const any = state.picked.some(Boolean);
      if (state.hover < 0) return any ? state.picked : state.picked.map(() => true);
      const sel = any ? state.picked.slice() : state.picked.map(() => false);
      sel[state.hover] = true;
      return sel;
    };

    const off = document.createElement('canvas');
    const photo = (canvas, img, dim = 0.38) => {
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.fillStyle = `rgba(10, 16, 12, ${dim})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };
    const paint = (canvas, values, n, { smooth = false, strength = 1.6, cap = 1, soft = false } = {}) => {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!values) return;
      off.width = n[1]; off.height = n[0];
      const octx = off.getContext('2d'), px = octx.createImageData(n[1], n[0]);
      for (let i = 0; i < values.length; i++) {
        const t = clamp(values[i], 0, 1), rgb = hot(t);
        px.data[i * 4] = rgb[0] * 255; px.data[i * 4 + 1] = rgb[1] * 255; px.data[i * 4 + 2] = rgb[2] * 255;
        px.data[i * 4 + 3] = Math.min(cap, (soft ? Math.sqrt(t) : t) * strength) * 255;
      }
      octx.putImageData(px, 0, 0);
      ctx.imageSmoothingEnabled = smooth;
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    };

    let layers = [], token = 0, queued = 0;
    const build = (meta) => {
      const names = meta.levels.length > 8 ? SHOWN : meta.levels.map((l) => l.name).filter((n) => n !== 'features');
      lanes.replaceChildren();
      layers = names.map((name) => {
        const k = meta.levels.findIndex((l) => l.name === name);
        const last = name === 'pixels', size = last ? 448 : 336;
        const base = layer('', size), canvas = layer('lrp-heat', size);
        const plane = h('div', { class: `lrp-plane${last ? ' lrp-plane--flat' : ''}` }, base, canvas);
        const num = h('b');
        lanes.append(h('div', { class: `lrp-layer${last ? ' lrp-layer--last' : ''}` },
          h('p', { class: 'lrp-layer-top', text: last ? 'where it lands' : '' }),
          h('div', { class: 'lrp-plane-box' }, plane),
          h('p', { class: 'lrp-layer-name' }, LEVEL(name)[0], num)));
        return { k, lv: meta.levels[k], base, canvas, plane, num, last };
      });
    };

    // the projected left and right edges of a tilted plane, in the coordinates of the layer wrapper
    const edges = (el, flat) => {
      const box = el.parentElement.getBoundingClientRect(), wrap = lanes.parentElement.getBoundingClientRect();
      const w = el.offsetWidth, cx = box.left - wrap.left + box.width / 2, cy = box.top - wrap.top + box.height / 2;
      const side = (sx) => {
        if (flat) return { x: cx + sx * w / 2, half: w / 2 };
        const x = sx * w / 2, z = -x * Math.sin(TILT), f = DEPTH / (DEPTH - z);
        return { x: cx + x * Math.cos(TILT) * f, half: (w / 2) * f };
      };
      return { left: side(-1), right: side(1), cy };
    };
    const drawRibbons = (amounts) => {
      ribbons.replaceChildren();
      if (getComputedStyle(ribbons).display === 'none' || !layers.length) return;
      const wrap = lanes.parentElement.getBoundingClientRect(), s = startBox.getBoundingClientRect();
      ribbons.setAttribute('viewBox', `0 0 ${wrap.width} ${wrap.height}`);
      const top = Math.max(1, ...amounts);
      const startEdge = { x: s.right - wrap.left, half: s.height / 2, cy: s.top - wrap.top + s.height / 2 };
      const ends = [startEdge, ...layers.map((l) => edges(l.plane, l.last))];
      const defs = svg('defs'), grad = svg('linearGradient', { id: 'lrp-ribbon', x1: 0, x2: 1, y1: 0, y2: 0 });
      grad.append(svg('stop', { offset: 0, 'stop-color': '#e2552b', 'stop-opacity': 0.5 }), svg('stop', { offset: 1, 'stop-color': '#f2a33a', 'stop-opacity': 0.5 }));
      defs.append(grad);
      ribbons.append(defs);
      for (let i = 0; i < ends.length - 1; i++) {
        const a = i === 0 ? startEdge : ends[i].right, b = i === 0 ? ends[1].left : ends[i + 1].left;
        const ay = i === 0 ? startEdge.cy : ends[i].cy, by = ends[i + 1].cy;
        const ha = a.half * 0.9 * clamp(amounts[i] / top, 0, 1), hb = b.half * 0.9 * clamp(amounts[i + 1] / top, 0, 1);
        const mx = (a.x + b.x) / 2;
        ribbons.append(svg('path', {
          class: 'lrp-ribbon',
          d: `M${a.x},${ay - ha} C${mx},${ay - ha} ${mx},${by - hb} ${b.x},${by - hb} L${b.x},${by + hb} C${mx},${by + hb} ${mx},${ay + ha} ${a.x},${ay + ha} Z`,
        }));
      }
    };

    const draw = async () => {
      const my = ++token;
      let meta, img;
      try {
        [meta, img] = await Promise.all([load(state.model, state.item), image(state.item)]);
      } catch (e) {
        status.textContent = 'The precomputed relevance could not be loaded.';
        return;
      }
      if (my !== token) return;
      if (state.data !== meta) {
        state.data = meta; state.img = img; state.hover = -1;
        state.picked = new Array(meta.cells).fill(false);
        build(meta);
        photo(startBase, img);
        layers.forEach((l) => photo(l.base, img));
      }
      const sel = selection(), all = sel.map(() => true), lv0 = meta.levels[0], g = meta.grid;

      // the start: positive features per cell (min to max), with the selection
      const start = toGrid(sumLevel(lv0, all), lv0.size, g);
      let lo = Infinity, hi = -Infinity;
      for (const v of start) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      paint(startHeat, start.map((v) => (v - lo) / (hi - lo || 1)), g, { strength: 1, cap: 0.75 });
      const ctx = startMarks.getContext('2d'), cw = startMarks.width / g[1], ch = startMarks.height / g[0];
      ctx.clearRect(0, 0, startMarks.width, startMarks.height);
      const any = state.picked.some(Boolean), few = state.picked.filter(Boolean).length <= 12;
      for (let r = 0; r < g[0]; r++) {
        for (let c = 0; c < g[1]; c++) {
          const i = r * g[1] + c, pick = state.picked[i], hov = i === state.hover;
          if (any && !pick && !hov) { ctx.fillStyle = 'rgba(10, 16, 12, 0.6)'; ctx.fillRect(c * cw, r * ch, cw, ch); }
          ctx.strokeStyle = pick ? (few ? '#fff' : 'rgba(255, 255, 255, 0.45)') : hov ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.2)';
          ctx.lineWidth = pick ? (few ? 5 : 2) : hov ? 3 : 1.5;
          ctx.setLineDash(hov && !pick ? [8, 6] : []);
          ctx.strokeRect(c * cw + 2.5, r * ch + 2.5, cw - 5, ch - 5);
        }
      }
      ctx.setLineDash([]);

      // the layers: exact maps, each scaled to its maximum or all on one scale (relevance per image area)
      const raw = layers.map((l) => {
        const area = l.lv.size[0] * l.lv.size[1];
        return resample(sumLevel(l.lv, sel), l.lv.size).map((v) => Math.max(0, v * area));
      });
      let refs = raw.map((v) => { let p = 0; for (const x of v) p = Math.max(p, x); return p || 1; });
      if (state.common) {
        const p99 = raw.map((v) => { const s = Float32Array.from(v).sort(); return s[Math.floor(0.99 * (s.length - 1))]; });
        refs = refs.map(() => Math.max(...p99) || 1);
      }
      const startAmount = amount(lv0, sel);
      const amounts = [1, ...layers.map((l) => amount(l.lv, sel) / startAmount)];
      layers.forEach((l, i) => {
        paint(l.canvas, raw[i].map((v) => v / refs[i]), [M, M], { smooth: l.last, soft: true, strength: 1.25 });
        l.num.textContent = fmt(amounts[i + 1]);
      });
      drawRibbons(amounts);

      const n = sel.filter(Boolean).length, share = startAmount / amount(lv0, all);
      const arrive = fmt(amounts[amounts.length - 1]);
      readout.textContent = n === sel.length
        ? `All cells start with the whole amount of 1.00, and ${arrive} of it reaches the pixels. The rest stays with biases and other learned constants on the way.`
        : `${n === 1 ? 'This cell holds' : `These ${n} cells hold`} ${fmt(share)} of the starting amount. Of that, ${arrive} reaches the pixels.`;
    };
    const render = () => { if (!queued) queued = requestAnimationFrame(() => { queued = 0; draw(); }); };

    const cellAt = (e) => {
      if (!state.data) return -1;
      const r = startBox.getBoundingClientRect(), [gr, gc] = state.data.grid;
      const c = Math.floor((e.clientX - r.left) / r.width * gc), q = Math.floor((e.clientY - r.top) / r.height * gr);
      return c >= 0 && c < gc && q >= 0 && q < gr ? q * gc + c : -1;
    };
    startBox.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return;
      const c = cellAt(e);
      if (c !== state.hover) { state.hover = c; render(); }
    });
    startBox.addEventListener('pointerleave', () => { if (state.hover >= 0) { state.hover = -1; render(); } });
    startBox.addEventListener('click', (e) => {
      const c = cellAt(e);
      if (c < 0 || !state.picked) return;
      state.picked[c] = !state.picked[c];
      render();
    });
    // nothing picked means all cells, so inverting all cells comes back to the full start
    invertBtn.addEventListener('click', () => {
      if (!state.picked) return;
      const any = state.picked.some(Boolean);
      state.picked = any ? state.picked.map((p) => !p) : state.picked.map(() => false);
      if (state.picked.every(Boolean)) state.picked.fill(false);
      render();
    });
    const setOpacity = (v) => {
      opacity.value = v; opacityOut.textContent = `${v}%`;
      root.style.setProperty('--lrp-heat-opacity', v / 100);
    };
    opacity.addEventListener('input', () => setOpacity(Number(opacity.value)));
    resetBtn.addEventListener('click', () => {
      if (state.picked) state.picked.fill(false);
      commonBox.checked = false; state.common = false;
      setOpacity(100);
      render();
    });
    commonBox.addEventListener('change', () => { state.common = commonBox.checked; render(); });
    modelSelect.addEventListener('change', () => { state.model = modelSelect.value; render(); });
    // the dock belongs to the shared gallery script; follow its selection
    new MutationObserver(() => {
      const on = thumbs.find((t) => t.getAttribute('aria-selected') === 'true');
      if (on && on.dataset.item !== state.item) { state.item = on.dataset.item; render(); }
    }).observe(root.querySelector('.fg-dock'), { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
    addEventListener('resize', () => { if (state.data) render(); }, { passive: true });

    if (!('DecompressionStream' in window)) {
      status.textContent = 'This browser cannot unpack the precomputed relevance. A recent version of any major browser can.';
      return;
    }
    render();
  };

  const markColumns = (root) => root.querySelectorAll('.lrp-table').forEach((t) => {
    const cols = Number(getComputedStyle(t).getPropertyValue('--lrp-cols')) || 4;
    [...t.children].forEach((c, i) => c.classList.toggle('lrp-c0', i % cols === 0));
  });

  const init = () => {
    const toys = { activation: activationToy, layernorm: layerNormToy, attention: attentionToy, residual: residualToy };
    document.querySelectorAll('[data-lrp-toy]').forEach((el) => {
      if (el.dataset.ready) return;
      el.dataset.ready = 'true';
      el.replaceChildren();
      toys[el.dataset.lrpToy]?.(el);
      markColumns(el);
    });
    document.querySelectorAll('[data-lrp-stack]').forEach((el) => {
      if (el.dataset.ready) return;
      el.dataset.ready = 'true';
      el.replaceChildren();
      stackExplorer(el.closest('figure'), el);
    });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

/* Interactive figures for the LRP post. The stacked layers read precomputed relevance; the neuron computes in the page. */
(() => {
  'use strict';

  /* element builders: h(tag, attrs, ...children) for HTML, svg(...) for SVG; on* attributes become listeners */
  const make = (ns) => (tag, attrs = {}, ...kids) => {
    const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    node.append(...kids.flat(Infinity).filter((c) => c != null && c !== false));
    return node;
  };
  const h = make(), svg = make('http://www.w3.org/2000/svg');
  const fmt = (v, digits = 2) => {
    if (!Number.isFinite(v)) return '∞';
    const s = v.toFixed(digits);
    return (Number(s) === 0 ? (0).toFixed(digits) : s).replace('-', '−');
  };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* matplotlib's "hot" colormap */
  const hot = (t) => [t / 0.365, (t - 0.365) / 0.381, (t - 0.746) / 0.254].map((c) => clamp(c, 0, 1));

  /* -------------------------------------------------------------------------------- neuron toy */
  const neuronToy = (root) => {
    const v = { x1: 2, x2: 1, w1: 1.5, w2: -1 };
    const LIM = { x1: [-2, 3], x2: [-2, 3], w1: [-2, 2], w2: [-2, 2] };
    const MODES = { grad: 'Gradient fallback', zplus: 'z⁺ rule', lrp0: 'LRP-0' };
    const RN = 40, RB = 30, UNIT = 7, SUB = '₁₂';
    const GEOM = {
      wide: { W: 840, H: 262, vertical: false, n: [470, 142], fin: [[62, 92], [62, 192]], bin: [[724, 92], [724, 192]],
        heads: [[0, 20, 'start', 'FORWARD · THE NEURON COMPUTES'], [840, 20, 'end', 'BACKWARD · RELEVANCE FLOWS BACK']] },
      tall: { W: 420, H: 600, vertical: true, n: [210, 300], fin: [[105, 70], [315, 70]], bin: [[105, 520], [315, 520]],
        heads: [[0, 16, 'start', 'FORWARD'], [0, 380, 'start', 'BACKWARD']] },
    };
    let mode = 'grad', S = null, layout = '';

    const bez = (p, t) => [0, 1].map((k) =>
      (1 - t) ** 3 * p[0][k] + 3 * (1 - t) ** 2 * t * p[1][k] + 3 * (1 - t) * t ** 2 * p[2][k] + t ** 3 * p[3][k]);

    const build = (name) => {
      const { W, H, vertical: vert, n: N, fin, bin, heads } = GEOM[name];
      const scene = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: `lrp-neuron-scene lrp-nt-${name}`, role: 'group',
        'aria-label': 'A neuron with two inputs. On one side the inputs compute the output, on the other side the output flows back to the inputs as relevance.' });
      const [gTrack, gStream, gFlow, gNodes, gText] = [0, 0, 0, 0, 0].map(() => scene.appendChild(svg('g')));
      const text = (x, y, cls, s = '', anchor = 'middle') => gText.appendChild(svg('text', { x, y, class: cls, 'text-anchor': anchor }, s));
      const path = (p, g, cls) => g.appendChild(svg('path', { d: `M${p[0]}C${p[1]} ${p[2]} ${p[3]}`, class: cls }));
      // the point on the rim of a node (radius r) that faces t, and the S-curve from a node to the neuron
      const rim = ([cx, cy], [tx, ty], r) => (vert ? [cx, cy + Math.sign(ty - cy) * r] : [cx + Math.sign(tx - cx) * r, cy]);
      const curve = (c, r) => {
        const a = rim(c, N, r), b = rim(N, c, RN), s = Math.sign(vert ? b[1] - a[1] : b[0] - a[0]) * 90;
        const [dx, dy] = vert ? [0, s] : [s, 0];
        return [a, [a[0] + dx, a[1] + dy], [b[0] - dx, b[1] - dy], b];
      };

      // a value chip: drag sideways or use the arrow keys
      const chip = (key, cx, cy, show) => {
        const t = svg('text', { x: cx, y: cy + 7, 'text-anchor': 'middle' });
        const arrow = (dx, s) => svg('text', { x: cx + dx, y: cy + 6, 'text-anchor': 'middle', class: 'lrp-nt-arrow' }, s);
        const set = (q) => { v[key] = clamp(Math.round(q * 10) / 10, ...LIM[key]); update(); };
        const end = () => node.classList.remove('lrp-nt-drag');
        let x0 = 0, v0 = 0, scale = 1;
        const node = gText.appendChild(svg('g', {
          class: 'lrp-nt-chip', tabindex: 0, role: 'slider', 'aria-label': key[0] + SUB[key[1] - 1],
          'aria-valuemin': LIM[key][0], 'aria-valuemax': LIM[key][1],
          onpointerdown: (e) => {
            node.setPointerCapture(e.pointerId);
            x0 = e.clientX; v0 = v[key]; scale = W / scene.getBoundingClientRect().width;
            node.classList.add('lrp-nt-drag'); root.classList.add('lrp-nt-touched');
          },
          onpointermove: (e) => { if (node.hasPointerCapture(e.pointerId)) set(v0 + (e.clientX - x0) * scale / 40); },
          onpointerup: end,
          onpointercancel: end,
          onkeydown: (e) => {
            const s = { ArrowRight: 0.1, ArrowUp: 0.1, ArrowLeft: -0.1, ArrowDown: -0.1, PageUp: 1, PageDown: -1 }[e.key];
            if (s) { e.preventDefault(); set(v[key] + s); root.classList.add('lrp-nt-touched'); }
          },
        }, svg('rect', { x: cx - 48, y: cy - 18, width: 96, height: 36, rx: 18 }), arrow(-37, '‹'), t, arrow(37, '›')));
        return () => { t.textContent = show(v[key]); node.setAttribute('aria-valuenow', v[key]); };
      };

      heads.forEach(([x, y, a, s]) => text(x, y, 'lrp-nt-head', s, a));
      const fwd = fin.map((c, i) => {
        const p = curve(c, vert ? 18 : 44), [x, y] = bez(p, 0.86);
        return {
          p, band: path(p, gStream, 'lrp-nt-fband'), flow: path(p, gFlow, 'lrp-nt-fflow'),
          label: vert ? text(x + (i ? 14 : -14), y, 'lrp-nt-small', '', i ? 'start' : 'end') : text(x, i ? y + 30 : y - 16, 'lrp-nt-small'),
        };
      });
      // backward inputs: the name next to the node, the relevance it receives inside it
      const back = bin.map(([x, y], i) => {
        const p = curve([x, y], RB);
        path(p, gTrack, 'lrp-nt-track');
        text(x, vert || i ? y + RB + 24 : y - RB - 12, 'lrp-nt-name', `x${SUB[i]}`);
        return { stream: path(p, gStream, 'lrp-nt-stream'), flow: path(p, gFlow, 'lrp-nt-flow'), val: text(x, y + 6, 'lrp-nt-gets') };
      });
      gNodes.append(svg('circle', { cx: N[0], cy: N[1], r: RN, class: 'lrp-nt-neuron' }),
        ...bin.map(([cx, cy]) => svg('circle', { cx, cy, r: RB, class: 'lrp-nt-input' })));
      const chips = [
        ...fin.map(([x, y], i) => {
          text(x, !vert && i ? y + 42 : y - 28, 'lrp-nt-name', `x${SUB[i]}`);
          return chip(`x${i + 1}`, x, y, (q) => fmt(q, 1));
        }),
        ...fwd.map((e, i) => {
          const [x, y] = bez(e.p, vert ? 0.42 : 0.4);
          if (vert) text(x + (i ? 58 : -58), y + 7, 'lrp-nt-name', `w${SUB[i]}`, i ? 'start' : 'end');
          else text(x, i ? y + 42 : y - 28, 'lrp-nt-name', `w${SUB[i]}`);
          return chip(`w${i + 1}`, x, y, (q) => `× ${fmt(q, 1)}`);
        }),
      ];
      return {
        scene, fwd, back, chips,
        out: text(N[0], N[1] + 9, 'lrp-nt-out'),
        outLbl: vert ? text(N[0] + RN + 10, N[1] - 6, 'lrp-nt-small', '', 'start') : text(N[0], N[1] + RN + 26, 'lrp-nt-small'),
        total: vert ? text(N[0], bin[0][1] + 7, 'lrp-nt-total') : text(bin[0][0], N[1] + 6, 'lrp-nt-total'),
      };
    };

    // stream width follows the amount; the moving dashes only show on streams wide enough to carry them
    const thick = (band, flow, q, max, min = 0) => {
      const w = Math.min(Math.abs(q) * UNIT, max);
      band.style.strokeWidth = `${Math.max(w, min)}px`;
      flow.style.strokeWidth = `${Math.min(2.6, w * 0.35)}px`;
      flow.style.opacity = w > 2.2 ? '' : '0';
    };
    const signed = (q, gap = '') => `${q < 0 ? '−' : '+'}${gap}${fmt(Math.abs(q), 1)}`;
    const f1 = (q) => (q < 0 ? `(${fmt(q, 1)})` : fmt(q, 1));

    const update = () => {
      const z = [v.x1 * v.w1, v.x2 * v.w2], y = z[0] + z[1];
      const pos = Math.max(0, z[0]) + Math.max(0, z[1]);
      const R = mode === 'zplus' ? Math.max(0, y) : y;
      const r = mode === 'grad' ? [v.w1 * y, v.w2 * y]
        : mode === 'lrp0' ? z
        : z.map((q) => (pos > 0 ? Math.max(0, q) / pos * R : 0));
      const total = r[0] + r[1], leak = Math.abs(total - R) > 0.005;

      S.chips.forEach((render) => render());
      S.fwd.forEach((e, i) => { thick(e.band, e.flow, z[i], 32, 1.5); e.label.textContent = signed(z[i]); });
      S.back.forEach((e, i) => {
        thick(e.stream, e.flow, r[i], 44);
        e.stream.classList.toggle('lrp-nt-neg', r[i] < 0);
        e.val.textContent = fmt(r[i], 1);
        e.val.classList.toggle('lrp-nt-neg', r[i] < -0.005);
        e.val.classList.toggle('lrp-nt-pos', r[i] > 0.005);
      });
      S.out.textContent = fmt(y, 1);
      S.outLbl.textContent = mode === 'zplus' && y <= 0 ? 'output, nothing flows back' : 'output';
      S.total.textContent = `total ${fmt(total, 1)}`;
      S.total.classList.toggle('lrp-nt-warn', leak);
      buttons.forEach((b) => b.setAttribute('aria-pressed', b.value === mode));
      eq.textContent = `${f1(v.x1)} × ${f1(v.w1)} + ${f1(v.x2)} × ${f1(v.w2)} = ${fmt(z[0], 1)} ${signed(z[1], ' ')} = ${fmt(y, 1)}`;
      note.textContent =
        mode === 'grad'
          ? (v.w1 === 1 && v.w2 === 1 ? 'With two weights of 1, as at a residual connection, the gradient hands the full amount to both inputs, and the total doubles.'
            : leak ? `The gradient hands each input its weight times the output, whatever the input contributed, so the inputs hold ${fmt(total, 1)} in total instead of the output of ${fmt(R, 1)}.`
            : 'The gradient hands each input its weight times the output. Here the weights add up to 1, so the total happens to match. Change one and it drifts.')
        : mode === 'zplus' && y <= 0 ? `The output is ${y < 0 ? 'negative' : 'zero'}, so there is no positive relevance to divide, and the z⁺ rule starts with 0, as in the first figure. Even an input that raised the output gets nothing. In a network, a ReLU would set such a neuron to zero anyway.`
        : v.x1 === 0 || v.x2 === 0 ? 'An input of zero contributes nothing and receives nothing, however large its weight.'
        : z.some((q) => q < 0) ? (mode === 'lrp0' ? 'One input works against the output and receives its negative contribution. The inputs still hold exactly the output.' : `The z⁺ rule ignores the input that works against the output and gives all of the ${fmt(R, 1)} to the other one.`)
        : 'Each input receives exactly what it contributed, so the inputs hold the whole output and nothing more.';
    };

    const buttons = Object.entries(MODES).map(([k, label]) => h('button', { type: 'button', value: k, text: label, onclick: () => { mode = k; update(); } }));
    const holder = h('div', { class: 'lrp-nt-holder' });
    const eq = h('output', { class: 'lrp-nt-eq' });
    const note = h('p', { class: 'lrp-small lrp-nt-note', 'aria-live': 'polite' });
    root.append(h('div', {},
      holder,
      h('div', { class: 'lrp-nt-bar' },
        h('p', { class: 'lrp-nt-eqline' }, 'output = ', eq),
        h('div', { class: 'lrp-buttons lrp-nt-modes', role: 'group', 'aria-label': 'How the relevance is passed back' }, buttons)),
      note));

    const fit = () => {
      const name = root.clientWidth >= 560 ? 'wide' : 'tall';
      if (name === layout) return;
      layout = name;
      S = build(name);
      holder.replaceChildren(S.scene);
      update();
    };
    fit();
    new ResizeObserver(fit).observe(root);
    new IntersectionObserver(([en]) => root.classList.toggle('lrp-nt-paused', !en.isIntersecting)).observe(root);
  };

  /* -------------------------------------------------------------------------------- stack explorer */
  /* where a level from the exported data sits, for its caption */
  const where = (name) => (name === 'pixels' ? 'the pixels' : name.replace(/^.*→ (?:stage )?(\d+)$/, 'below stage $1'));
  const SHOWN = ['below the final norm', 'below block 9', 'below block 5', 'below block 1', 'pixels'];
  const OPACITY = 70;                               // default map opacity, in percent
  const M = 112;                                    // display resolution of the layer maps
  const GAP = 40, BIG = 1.6, GROW = 1.4, NEAR = 0.3; // space for the bands; end panels vs. the depths; how much a hovered depth and its neighbours grow
  const narrow = window.matchMedia('(max-width: 640px)');
  const calm = window.matchMedia('(prefers-reduced-motion: reduce)');

  const stackExplorer = (root, app) => {
    const models = JSON.parse(app.dataset.models);
    const thumbs = [...root.querySelectorAll('.fg-thumb')];
    const state = { model: models[0].id, item: thumbs[0]?.dataset.item, data: null, hover: -1, picked: null, common: false, amounts: null };

    /* data, fetched once per model and image: int8 maps per level and start cell, with one scale per cell */
    const cache = new Map();
    const once = (key, fn) => { if (!cache.has(key)) cache.set(key, fn()); return cache.get(key); };
    const load = (key) => once(key, async () => {
      const [meta, buf] = await Promise.all([
        fetch(`${app.dataset.base}${key}.json`).then((r) => r.json()),
        fetch(`${app.dataset.base}${key}.dat`).then((r) => new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer()),
      ]);
      const q = new Int8Array(buf);
      meta.cells = meta.grid[0] * meta.grid[1];
      let o = 0;
      for (const lv of meta.levels) lv.q = q.subarray(o, (o += meta.cells * lv.size[0] * lv.size[1]));
      return meta;
    });
    const image = (item) => once(`img:${item}`, () => new Promise((resolve, reject) => Object.assign(new Image(), {
      onload() { resolve(this); }, onerror: reject, src: thumbs.find((t) => t.dataset.item === item).dataset.src,
    })));

    /* controls */
    const setOpacity = (x) => {
      opacity.value = x; opacityOut.textContent = `${x}%`;
      root.style.setProperty('--lrp-heat-opacity', x / 100);
    };
    // nothing picked means all cells, so inverting all cells comes back to the full start
    const invert = () => {
      if (!state.picked) return;
      const any = state.picked.some(Boolean);
      state.picked = state.picked.map((p) => any && !p);
      if (state.picked.every(Boolean)) state.picked.fill(false);
      render();
    };
    const reset = () => {
      state.picked?.fill(false);
      commonBox.checked = state.common = false;
      setOpacity(OPACITY);
      render();
    };
    const hover = (c) => { if (c !== state.hover) { state.hover = c; render(); } };
    const cellAt = (e) => {
      if (!state.data) return -1;
      const r = startBox.getBoundingClientRect(), [gr, gc] = state.data.grid;
      const c = Math.floor((e.clientX - r.left) / r.width * gc), q = Math.floor((e.clientY - r.top) / r.height * gr);
      return c >= 0 && c < gc && q >= 0 && q < gr ? q * gc + c : -1;
    };
    const icon = (label, onclick, ...paths) => h('button', { type: 'button', class: 'lrp-icon', title: label, 'aria-label': label, onclick },
      svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' }, paths.map(([d, fill]) => svg('path', { d, class: fill && 'lrp-icon-fill' }))));
    const canvas = (heat) => h('canvas', { class: heat ? 'lrp-canvas lrp-heat' : 'lrp-canvas', width: 448, height: 448 });

    const modelSelect = h('select', { 'aria-label': 'Model', onchange: () => { state.model = modelSelect.value; render(); } },
      models.map((m) => h('option', { value: m.id, text: m.label })));
    const opacity = h('input', { type: 'range', min: 0, max: 100, step: 5, value: OPACITY, 'aria-label': 'Map opacity', oninput: () => setOpacity(Number(opacity.value)) });
    const opacityOut = h('output');
    const commonBox = h('input', { type: 'checkbox', onchange: () => { state.common = commonBox.checked; render(); } });
    const startBase = canvas(), startHeat = canvas(true), startMarks = canvas();
    const startBox = h('div', {
      class: 'lrp-start', role: 'img', 'aria-label': 'The start. Hover a cell to preview it, click cells to select them.',
      onpointermove: (e) => { if (e.pointerType === 'mouse') hover(cellAt(e)); },
      onpointerleave: () => hover(-1),
      onclick: (e) => { const c = cellAt(e); if (c >= 0) { state.picked[c] = !state.picked[c]; render(); } },
    }, startBase, startHeat, startMarks);
    const tools = h('div', { class: 'lrp-tools', role: 'toolbar', 'aria-label': 'Start cells' },
      icon('Invert the selection', invert, ['M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16z'], ['M12 4a8 8 0 0 1 0 16z', true]),
      icon('Reset: all cells, default scale and opacity', reset, ['M4 12a8 8 0 1 0 2.4-5.7'], ['M4 4v4.5h4.5']));
    const startCol = h('div', { class: 'lrp-col lrp-col--start' },
      h('p', { class: 'lrp-layer-top', text: 'where it starts' }), startBox, h('p', { class: 'lrp-layer-name' }, 'the features'), tools);
    const ribbons = svg('svg', { class: 'lrp-ribbons', 'aria-hidden': 'true' });
    const row = h('div', {
      class: 'lrp-row',
      onpointermove: (e) => {
        if (e.pointerType !== 'mouse') return;
        if (startCol.contains(e.target) || layers.at(-1)?.col.contains(e.target)) return zoomTo(-1);
        const j = mids().findIndex((l) => { const r = l.col.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right; });
        if (j >= 0) zoomTo(j);                        // in a gap keep the current one, so the row does not flicker
      },
      onpointerleave: () => zoomTo(-1),
    }, ribbons, startCol);
    const readout = h('p', { class: 'lrp-result', 'aria-live': 'polite' });
    const status = h('p', { class: 'lrp-small lrp-status' });
    app.append(
      h('div', { class: 'lrp-stack-controls' },
        h('label', { class: 'lrp-select' }, 'Model', modelSelect),
        h('label', { class: 'lrp-range' }, 'Map opacity', opacity, opacityOut),
        h('label', { class: 'lrp-check' }, commonBox, ' One colour scale for all depths')),
      row, readout, status);

    /* relevance arithmetic */
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
        const o = Math.min(r - 1, (y * r / M) | 0) * c;
        for (let x = 0; x < M; x++) out[y * M + x] = v[o + Math.min(c - 1, (x * c / M) | 0)];
      }
      return out;
    };
    const selection = () => {                       // clicked cells plus the hovered one; nothing clicked = all
      const any = state.picked.some(Boolean);
      if (state.hover < 0) return any ? state.picked : state.picked.map(() => true);
      const sel = state.picked.map((p) => any && p);
      sel[state.hover] = true;
      return sel;
    };

    /* drawing */
    const off = document.createElement('canvas');
    const photo = (c, img) => c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    // The maps are opaque, so the opacity slider blends between the plain photo (0) and the map alone (100).
    // soft: colour by the square root, so that the weak spread of the plain ViTs stays visible.
    const heat = (target, values, [rows, cols], { smooth = false, soft = false } = {}) => {
      off.width = cols; off.height = rows;
      const octx = off.getContext('2d'), px = octx.createImageData(cols, rows);
      values.forEach((x, i) => {
        const t = clamp(x, 0, 1), [r, g, b] = hot(soft ? Math.sqrt(t) : t);
        px.data.set([r * 255, g * 255, b * 255, 255], i * 4);
      });
      octx.putImageData(px, 0, 0);
      const ctx = target.getContext('2d');
      ctx.imageSmoothingEnabled = smooth;
      ctx.drawImage(off, 0, 0, target.width, target.height);
    };

    let layers = [], token = 0, queued = 0;
    const build = (meta) => {
      const names = meta.levels.length > 8 ? SHOWN : meta.levels.map((l) => l.name).filter((n) => n !== 'features');
      layers.forEach((l) => l.col.remove());
      layers = names.map((name) => {
        const last = name === 'pixels', base = canvas(), map = canvas(true), num = h('b');
        const plane = h('div', { class: 'lrp-plane' }, base, map);
        const col = h('div', { class: 'lrp-col lrp-col--layer' },
          h('p', { class: 'lrp-layer-top', text: last ? 'where it lands' : '' }), plane, h('p', { class: 'lrp-layer-name' }, where(name), num));
        row.append(col);
        return { lv: meta.levels.find((l) => l.name === name), base, map, plane, num, col, last };
      });
      relayout();
    };

    // The start and the pixels are the large panels at both ends. The depths between them are smaller, and the one
    // under the pointer grows like a thumbnail in the dock, its neighbours a little, while the others give up the
    // space, so the row keeps its width.
    const fish = { zoom: -1, sizes: [], target: [], s0: 0, frame: 0 };
    const mids = () => layers.slice(0, -1);
    const sizesFor = (zoom, n) => {
      const w = Array.from({ length: n }, (_, i) => 1 + GROW * (zoom < 0 ? 0 : i === zoom ? 1 : Math.abs(i - zoom) === 1 ? NEAR : 0));
      const k = n * fish.s0 / w.reduce((a, b) => a + b, 0);
      return w.map((x) => x * k);
    };
    const apply = () => {
      startCol.style.width = layers.at(-1).col.style.width = `${BIG * fish.s0}px`;
      mids().forEach((l, i) => { l.col.style.width = `${fish.sizes[i]}px`; });
      drawRibbons();
    };
    const step = () => {
      let moving = false;
      fish.sizes = fish.sizes.map((s, i) => {
        const t = fish.target[i], next = calm.matches ? t : s + (t - s) * 0.24;
        if (Math.abs(t - next) < 0.4) return t;
        moving = true;
        return next;
      });
      apply();
      fish.frame = moving ? requestAnimationFrame(step) : 0;
    };
    const layout = () => {
      if (narrow.matches || !layers.length) {
        [startCol, ...layers.map((l) => l.col)].forEach((c) => { c.style.width = ''; });
        row.style.height = row.style.paddingBottom = '';
        fish.zoom = -1;
        return drawRibbons();
      }
      const n = layers.length - 1;
      // as wide as the row allows, but short enough that the panels and the controls fit on one screen
      fish.s0 = Math.min(0.28 * innerHeight, (row.clientWidth - (n + 1) * GAP) / (n + 2 * BIG));
      fish.target = sizesFor(fish.zoom, n);
      if (fish.sizes.length !== n) fish.sizes = fish.target.slice();
      // room for the largest panel any hover can produce (an end depth, which has one neighbour), so the page below never moves
      const top = Math.max(BIG * fish.s0, ...sizesFor(0, n));
      const labels = startCol.offsetHeight - startBox.offsetHeight;   // the toolbar hangs below, outside the column
      const pad = tools.offsetHeight + 14;
      row.style.paddingBottom = `${pad}px`;
      row.style.height = `${Math.ceil(top + labels + pad)}px`;
      apply();
    };
    const relayout = () => { fish.sizes = []; layout(); };
    const zoomTo = (j) => {
      if (j === fish.zoom || narrow.matches) return;
      fish.zoom = j;
      fish.target = sizesFor(j, mids().length);
      fish.frame ||= requestAnimationFrame(step);
    };

    // bands between the panels, as high as the amount that passes there (the same scale for all of them)
    const drawRibbons = () => {
      ribbons.replaceChildren();
      const a = state.amounts;
      if (!a || !layers.length || getComputedStyle(ribbons).display === 'none') return;
      const wrap = row.getBoundingClientRect();
      ribbons.setAttribute('viewBox', `0 0 ${wrap.width} ${wrap.height}`);
      const boxes = [startBox, ...layers.map((l) => l.plane)].map((el) => el.getBoundingClientRect());
      const k = 0.45 * Math.min(...boxes.map((b) => b.height)) / Math.max(1, ...a);
      ribbons.append(svg('defs', {}, svg('linearGradient', { id: 'lrp-ribbon', x1: 0, x2: 1, y1: 0, y2: 0 },
        svg('stop', { offset: 0, 'stop-color': '#e2552b', 'stop-opacity': 0.5 }),
        svg('stop', { offset: 1, 'stop-color': '#f2a33a', 'stop-opacity': 0.5 }))));
      for (let i = 1; i < boxes.length; i++) {
        const p = boxes[i - 1], q = boxes[i];
        const ax = p.right - wrap.left, bx = q.left - wrap.left, mx = (ax + bx) / 2;
        const ay = p.top + p.height / 2 - wrap.top, by = q.top + q.height / 2 - wrap.top;
        const ha = k * Math.max(0, a[i - 1]), hb = k * Math.max(0, a[i]);
        ribbons.append(svg('path', {
          class: 'lrp-ribbon',
          d: `M${ax},${ay - ha} C${mx},${ay - ha} ${mx},${by - hb} ${bx},${by - hb} L${bx},${by + hb} C${mx},${by + hb} ${mx},${ay + ha} ${ax},${ay + ha} Z`,
        }));
      }
    };

    const draw = async () => {
      const my = ++token;
      let meta, img;
      try {
        [meta, img] = await Promise.all([load(`${state.model}_${state.item}`), image(state.item)]);
      } catch {
        status.textContent = 'The precomputed relevance could not be loaded.';
        return;
      }
      if (my !== token) return;
      status.textContent = '';
      if (state.data !== meta) {
        Object.assign(state, { data: meta, hover: -1, picked: new Array(meta.cells).fill(false), amounts: null });
        build(meta);
        [startBase, ...layers.map((l) => l.base)].forEach((c) => photo(c, img));
      }
      const sel = selection(), all = sel.map(() => true), lv0 = meta.levels[0], gc = meta.grid[1];

      // the start: positive features per cell (min to max), with the selection
      const start = toGrid(sumLevel(lv0, all), lv0.size, meta.grid);
      const lo = Math.min(...start), hi = Math.max(...start);
      heat(startHeat, start.map((x) => (x - lo) / (hi - lo || 1)), meta.grid);
      const ctx = startMarks.getContext('2d'), cw = startMarks.width / gc, ch = startMarks.height / meta.grid[0];
      const any = state.picked.some(Boolean), few = state.picked.filter(Boolean).length <= 12;
      ctx.clearRect(0, 0, startMarks.width, startMarks.height);
      state.picked.forEach((pick, i) => {
        const x = (i % gc) * cw, y = Math.floor(i / gc) * ch, hov = i === state.hover;
        if (any && !pick && !hov) { ctx.fillStyle = 'rgba(10, 16, 12, 0.6)'; ctx.fillRect(x, y, cw, ch); }
        ctx.strokeStyle = pick ? (few ? '#fff' : 'rgba(255, 255, 255, 0.45)') : hov ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = pick ? (few ? 5 : 2) : hov ? 3 : 1.5;
        ctx.setLineDash(hov && !pick ? [8, 6] : []);
        ctx.strokeRect(x + 2.5, y + 2.5, cw - 5, ch - 5);
      });

      // the layers: exact maps, each scaled to its maximum or all on one scale (relevance per image area)
      const raw = layers.map(({ lv }) => {
        const area = lv.size[0] * lv.size[1];
        return resample(sumLevel(lv, sel), lv.size).map((x) => Math.max(0, x * area));
      });
      const p99 = (x) => x.slice().sort()[Math.floor(0.99 * (x.length - 1))];
      const common = state.common && (Math.max(...raw.map(p99)) || 1);
      const startAmount = amount(lv0, sel);
      const amounts = [1, ...layers.map((l) => amount(l.lv, sel) / startAmount)];
      layers.forEach((l, i) => {
        const ref = common || raw[i].reduce((m, x) => Math.max(m, x), 0) || 1;
        heat(l.map, raw[i].map((x) => x / ref), [M, M], { smooth: l.last, soft: true });
        l.num.textContent = fmt(amounts[i + 1]);
      });
      state.amounts = amounts;
      drawRibbons();

      const n = sel.filter(Boolean).length, arrive = fmt(amounts.at(-1));
      readout.textContent = n === sel.length
        ? `All cells start with the whole amount of 1.00, and ${arrive} of it reaches the pixels. The rest stays with biases and other learned constants on the way.`
        : `${n === 1 ? 'This cell holds' : `These ${n} cells hold`} ${fmt(startAmount / amount(lv0, all))} of the starting amount. Of that, ${arrive} reaches the pixels.`;
    };
    const render = () => { queued ||= requestAnimationFrame(() => { queued = 0; draw(); }); };

    // the dock belongs to the shared gallery script; follow its selection
    new MutationObserver(() => {
      const on = thumbs.find((t) => t.getAttribute('aria-selected') === 'true');
      if (on && on.dataset.item !== state.item) { state.item = on.dataset.item; render(); }
    }).observe(root.querySelector('.fg-dock'), { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
    addEventListener('resize', relayout, { passive: true });

    if (!('DecompressionStream' in window)) {
      status.textContent = 'This browser cannot unpack the precomputed relevance. A recent version of any major browser can.';
      return;
    }
    setOpacity(OPACITY);
    render();
  };

  const init = () => document.querySelectorAll('[data-lrp-toy], [data-lrp-stack]').forEach((el) => {
    if (el.dataset.ready) return;
    el.dataset.ready = 'true';
    el.replaceChildren();
    if ('lrpStack' in el.dataset) stackExplorer(el.closest('figure'), el);
    else if (el.dataset.lrpToy === 'neuron') neuronToy(el);
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

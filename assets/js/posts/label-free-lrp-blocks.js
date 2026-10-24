(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.append(e); return e; };
  const T = (parent, x, y, text, cls = 'lrpb-t', anchor = 'start') => { const e = S('text', { x, y, class: cls, 'text-anchor': anchor }, parent); e.textContent = text; return e; };
  const minus = (s) => String(s).replace(/-/g, '−');
  const num = (v, d = 1) => minus(v.toFixed(d));
  const pct = (v) => { const r = v * 100; return `${(r > 99 && r < 100) || (r > 0 && r < 1) ? r.toFixed(1) : Math.round(r)}%`; };
  const norm = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

  const BLOCKS = {
    convnext: (d) => ({
      title: 'ConvNeXt block', branches: [[
        { id: 'dw', label: '7 × 7 depthwise conv', rule: true, viz: 'window',
          text: 'Each channel is mixed with its own neighbours in a 7 × 7 window, one channel at a time. Below, the image cell and the window around it, then one channel before the convolution, its 7 × 7 weights, and the same channel after it.',
          status: 'A convolution with weights, so zennit’s rules for convolutions apply.' },
        { id: 'norm', label: 'LayerNorm', rule: false, viz: 'norm', from: 'dw', mean: d.norm_mean, std: d.norm_std,
          text: 'LayerNorm subtracts the mean of the cell’s values and divides them by their spread. It then multiplies each channel by a learned weight and adds a learned shift.',
          status: 'No rule in zennit. Without one, it splits the relevance into positive and negative parts that cancel. Our rule holds the spread fixed.' },
        { id: 'fc1', label: `linear ${d.channels} → ${d.fc1.length}`, rule: true, viz: 'fan', n: [d.channels, d.fc1.length],
          text: `Each of the ${d.fc1.length} outputs adds up all ${d.channels} values of the cell, each times its own weight, like the neuron above. ConvNeXt does this for every cell on its own.`,
          status: 'A layer with weights, covered by zennit.' },
        { id: 'gelu', label: 'GELU', rule: false, viz: 'gelu', from: 'fc1',
          text: `GELU bends every value on its own. Positive values pass almost unchanged, and negative ones are pushed towards zero. In this cell, ${pct(d.fc1.filter((v) => v < 0).length / d.fc1.length)} of the ${d.fc1.length} values are negative.`,
          status: 'zennit passes relevance through activations unchanged, but timm’s ConvNeXt uses its own GELU class, which zennit does not recognize. The relevance is then multiplied by the slope of the curve, as a gradient would be.' },
        { id: 'fc2', label: `linear ${d.fc1.length} → ${d.channels}`, rule: true, viz: 'fan', n: [d.fc1.length, d.channels],
          text: `Each of the ${d.channels} outputs adds up all ${d.fc1.length} values, each times its own weight.`,
          status: 'A layer with weights, covered by zennit.' },
        { id: 'ls', label: 'layer scale', rule: false, viz: 'scale',
          text: `Layer scale multiplies each channel of the block’s result by its own learned number. In this block, the numbers range from ${num(Math.min(...d.gamma))} to ${num(Math.max(...d.gamma))}.`,
          status: 'This is not a layer but a multiplication in the block’s code, so no rule can attach to it. Our canonizer turns it into a step that passes relevance on.' },
      ]], adds: [{ id: 'add', a: d.stream, b: d.update, what: 'The block’s result' }],
    }),
    vit: (d) => ({
      title: 'ViT block', branches: [[
        { id: 'norm1', label: 'LayerNorm', rule: false, viz: 'norm', from: 'stream', mean: d.norm1_mean, std: d.norm1_std,
          text: 'LayerNorm subtracts the mean of the token’s values and divides them by their spread. It then multiplies each channel by a learned weight and adds a learned shift.',
          status: 'No rule in zennit. Without one, it splits the relevance into positive and negative parts that cancel. Our rule holds the spread fixed.' },
        { id: 'qkv', label: 'query, key, value', rule: true, viz: 'fan', n: [d.channels, 3 * d.channels], groups: ['query', 'key', 'value'],
          text: `A linear layer turns each token’s ${d.channels} values into a query, a key and a value, ${d.channels} numbers each, split over ${d.heads} heads.`,
          status: 'A layer with weights, covered by zennit.' },
        { id: 'weights', label: 'attention weights', rule: false, viz: 'attention',
          text: `The query of each token is compared with the keys of all tokens, and a softmax turns the results into weights that add up to 1. The map shows the weights of one token, averaged over the ${d.heads} heads. Hover the image to pick another token.`,
          status: 'No rule in zennit. Without one, part of the relevance flows into the computation of the weights, in positive and negative amounts that cancel. We hold the weights fixed.' },
        { id: 'avg', label: 'weighted average', rule: false, viz: 'attention',
          text: 'Each token’s new value is the average of the values of all tokens, weighted with the attention weights. Tokens with a large weight contribute the most. Hover the image to pick another token.',
          status: 'No rule in zennit. With the weights held fixed, it divides relevance like a layer with weights.' },
        { id: 'proj', label: `linear ${d.channels} → ${d.channels}`, rule: true, viz: 'fan', n: [d.channels, d.channels],
          text: `A linear layer mixes the results of the ${d.heads} heads back into ${d.channels} values.`,
          status: 'A layer with weights, covered by zennit.' },
      ], [
        { id: 'norm2', label: 'LayerNorm', rule: false, viz: 'norm', from: 'mid', mean: d.norm2_mean, std: d.norm2_std,
          text: 'The second LayerNorm of the block, before its linear layers. It subtracts the mean of the token’s values, divides them by their spread, and applies a learned weight and shift to each channel.',
          status: 'No rule in zennit. Our rule holds the spread fixed.' },
        { id: 'fc1', label: `linear ${d.channels} → ${d.fc1.length}`, rule: true, viz: 'fan', n: [d.channels, d.fc1.length],
          text: `Each of the ${d.fc1.length} outputs adds up all ${d.channels} values of the token, each times its own weight. Unlike attention, this happens for every token on its own.`,
          status: 'A layer with weights, covered by zennit.' },
        { id: 'gelu', label: 'GELU', rule: true, viz: 'gelu', from: 'fc1',
          text: `GELU bends every value on its own. Positive values pass almost unchanged, and negative ones are pushed towards zero. In this token, ${pct(d.fc1.filter((v) => v < 0).length / d.fc1.length)} of the ${d.fc1.length} values are negative.`,
          status: 'timm’s ViT uses PyTorch’s own GELU, which zennit recognizes and passes relevance through unchanged.' },
        { id: 'fc2', label: `linear ${d.fc1.length} → ${d.channels}`, rule: true, viz: 'fan', n: [d.fc1.length, d.channels],
          text: `Each of the ${d.channels} outputs adds up all ${d.fc1.length} values, each times its own weight.`,
          status: 'A layer with weights, covered by zennit.' },
      ]], adds: [{ id: 'add1', a: d.stream, b: d.attn, what: 'The result of attention' }, { id: 'add2', a: d.mid, b: d.fc2, what: 'The result of the linear layers' }],
    }),
  };
  const ADD_STATUS = 'No rule in zennit for this block. Without one, the input and the result each receive the full relevance, so it doubles. Our canonizer splits it between them.';

  const BX = 84, BW = 168, BH = 28, STEP = 38, SX = 34, W = 270;

  const figure = async (root) => {
    const panel = root.querySelector('.lrpb-panel');
    let data;
    try { data = await (await fetch(root.dataset.src)).json(); } catch (e) { panel.textContent = 'The values could not be loaded.'; return; }
    const img = root.dataset.image, G = 14;
    const state = { model: null, part: null, query: data.row * G + data.col };
    const parts = new Map();

    const diagram = (host, key) => {
      const d = data[key], spec = BLOCKS[key](d), rows = spec.branches.reduce((n, b) => n + b.length, 0) + spec.adds.length;
      const H = 64 + rows * STEP + spec.branches.length * 26;
      const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${spec.title}, ${d.name}` }, host);
      T(svg, 0, 14, spec.title, 'lrpb-title');
      T(svg, 0, 30, `${d.name}, ${d.block}`, 'lrpb-sub');
      S('path', { class: 'lrpb-stream', d: `M${SX},40 V${H - 6}` }, svg);
      S('path', { class: 'lrpb-stream-head', d: `M${SX - 6},${H - 12} L${SX},${H - 2} L${SX + 6},${H - 12}` }, svg);
      let y = 58;
      spec.branches.forEach((branch, b) => {
        const top = y;
        S('path', { class: 'lrpb-wire', d: `M${SX},${top - 8} H${BX + BW / 2} V${top}` }, svg);
        branch.forEach((p, i) => {
          const g = S('g', { class: `lrpb-part${p.rule ? '' : ' lrpb-missing'}`, tabindex: 0, role: 'button', 'aria-label': `${p.label}, ${p.rule ? 'zennit has a rule' : 'no rule in zennit'}` }, svg);
          S('rect', { x: BX, y, width: BW, height: BH, rx: 6 }, g);
          T(g, BX + BW / 2, y + BH / 2 + 4.5, p.label, 'lrpb-label', 'middle');
          if (i < branch.length - 1) S('path', { class: 'lrpb-wire', d: `M${BX + BW / 2},${y + BH} V${y + STEP}` }, svg);
          bind(g, key, { ...p, model: key, d });
          y += STEP;
        });
        const add = spec.adds[b], ay = y + 6;
        S('path', { class: 'lrpb-wire', d: `M${BX + BW / 2},${y - STEP + BH} V${ay} H${SX + 12}` }, svg);
        const g = S('g', { class: 'lrpb-part lrpb-missing lrpb-add', tabindex: 0, role: 'button', 'aria-label': 'addition of the input and the result, no rule in zennit' }, svg);
        S('circle', { cx: SX, cy: ay, r: 12 }, g);
        T(g, SX, ay + 5.5, '+', 'lrpb-plus', 'middle');
        bind(g, key, { id: add.id, label: 'addition', rule: false, viz: 'add', a: add.a, b: add.b, what: add.what, model: key, d,
          text: `${add.what} is added to the input that went around it, channel by channel. Below are the 16 channels it changes most. Over all ${d.channels} channels, it is about ${pct(norm(add.b) / norm(add.a))} of the input’s size.`,
          status: ADD_STATUS });
        y = ay + 34;
      });
    };

    const bind = (g, key, p) => {
      parts.set(`${key}/${p.id}`, { g, p });
      const show = () => select(key, p.id);
      g.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') show(); });
      g.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') show(); });
      g.addEventListener('focus', show);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    };

    const select = (key, id) => {
      if (state.model === key && state.part === id) return;
      state.model = key; state.part = id;
      for (const [k, { g }] of parts) g.classList.toggle('lrpb-on', k === `${key}/${id}`);
      render();
    };

    const axis = (svg, x0, x1, y, lo, hi) => {
      S('line', { class: 'lrpb-axis', x1: x0, x2: x1, y1: y, y2: y }, svg);
      const X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0);
      for (const v of [lo, 0, hi]) {
        if (v < lo || v > hi) continue;
        S('line', { class: 'lrpb-axis', x1: X(v), x2: X(v), y1: y, y2: y + 4 }, svg);
        T(svg, X(v), y + 15, num(v, Math.abs(v) < 10 && v !== 0 ? 1 : 0), 'lrpb-tick', 'middle');
      }
      return X;
    };
    const strip = (svg, y, label, vals, mean, std, range) => {
      const [lo, hi] = range || [Math.min(...vals, mean - std), Math.max(...vals, mean + std)], pad = (hi - lo) * 0.04;
      T(svg, 0, y, label, 'lrpb-rowlabel');
      const X = axis(svg, 10, 410, y + 40, lo - pad, hi + pad);
      if (std) S('rect', { class: 'lrpb-band', x: X(mean - std), y: y + 8, width: X(mean + std) - X(mean - std), height: 30 }, svg);
      vals.forEach((v, i) => S('circle', { class: 'lrpb-dot', cx: X(v), cy: y + 14 + (i * 7919 % 19), r: 2 }, svg));
      if (std) {
        S('line', { class: 'lrpb-mean', x1: X(mean), x2: X(mean), y1: y + 4, y2: y + 42 }, svg);
        T(svg, 410, y, `mean ${num(mean, 2)}, spread ${num(std, 2)}`, 'lrpb-tick', 'end');
      }
    };
    const signedMap = (vals, n) => {
      const c = document.createElement('canvas'); c.width = c.height = n;
      const ctx = c.getContext('2d'), im = ctx.createImageData(n, n), m = Math.max(...vals.map(Math.abs)) || 1;
      const pos = [20, 104, 95], neg = [113, 81, 142];
      vals.forEach((v, i) => { const t = Math.min(1, Math.abs(v) / m), base = v >= 0 ? pos : neg; im.data.set([...base.map((b) => Math.round(255 + (b - 255) * t)), 255], i * 4); });
      ctx.putImageData(im, 0, 0); return c.toDataURL();
    };
    const square = (svg, x, y, s, href, n, label) => {
      S('image', { href, x, y, width: s, height: s, preserveAspectRatio: 'none', style: 'image-rendering: pixelated' }, svg);
      S('rect', { class: 'lrpb-frame', x, y, width: s, height: s }, svg);
      T(svg, x + s / 2, y + s + 16, label, 'lrpb-tick', 'middle');
      return (r, c, h = 1, w = 1) => S('rect', { class: 'lrpb-mark', x: x + c * s / n, y: y + r * s / n, width: w * s / n, height: h * s / n }, svg);
    };
    const weightsOf = (d) => {
      if (d._w) return d._w;
      const raw = atob(d.weights), n = d.weights_max.length, w = new Float32Array(n * n);
      for (let i = 0; i < n * n; i++) w[i] = raw.charCodeAt(i) / 255 * d.weights_max[(i / n) | 0];
      return (d._w = { w, n });
    };

    const viz = (svg, p) => {
      const d = p.d, r = data.row, c = data.col;
      if (p.viz === 'norm') {
        const vals = d[p.from], after = vals.map((v) => (v - p.mean) / p.std), all = [...vals, ...after], range = [Math.min(...all), Math.max(...all)];
        strip(svg, 14, `the ${d.channels} values it reads`, vals, p.mean, p.std, range);
        strip(svg, 104, 'minus the mean, divided by the spread', after, 0, 1, range);
        return 170;
      }
      if (p.viz === 'scale') { strip(svg, 14, `the ${d.channels} learned numbers`, d.gamma, 0, 0); return 80; }
      if (p.viz === 'gelu') {
        const xs = d[p.from], lo = Math.min(...xs, -1) - 0.3, hi = Math.max(...xs, 1) + 0.3, top = gelu(hi), x0 = 30, x1 = 410, y0 = 190, y1 = 12;
        const X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0), Y = (v) => y0 - (v + 0.3) / (top + 0.3) * (y0 - y1);
        S('line', { class: 'lrpb-axis', x1: x0, x2: x1, y1: Y(0), y2: Y(0) }, svg);
        S('line', { class: 'lrpb-axis', x1: X(0), x2: X(0), y1: y1, y2: y0 }, svg);
        let path = ''; for (let i = 0; i <= 120; i++) { const v = lo + (hi - lo) * i / 120; path += `${i ? 'L' : 'M'}${X(v).toFixed(1)},${Y(gelu(v)).toFixed(1)}`; }
        S('path', { class: 'lrpb-curve', d: path }, svg);
        xs.forEach((v) => S('circle', { class: 'lrpb-dot lrpb-dot--on', cx: X(v), cy: Y(gelu(v)), r: 2.2 }, svg));
        S('line', { class: 'lrpb-axis', x1: x0, x2: x1, y1: y0 + 22, y2: y0 + 22 }, svg);
        xs.forEach((v) => S('line', { class: 'lrpb-rug', x1: X(v), x2: X(v), y1: y0 + 10, y2: y0 + 22 }, svg));
        T(svg, x1, y0 + 38, `the ${xs.length} values going in, from ${num(Math.min(...xs))} to ${num(Math.max(...xs))}`, 'lrpb-tick', 'end');
        T(svg, X(0) + 6, y1 + 8, 'value coming out', 'lrpb-tick');
        return 235;
      }
      if (p.viz === 'fan') {
        const [a, b] = p.n, na = Math.min(8, a), nb = Math.min(12, b), ys = (k, n) => 20 + (k + 0.5) * 150 / n;
        for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++) S('line', { class: 'lrpb-fanline', x1: 130, y1: ys(i, na), x2: 290, y2: ys(j, nb) }, svg);
        for (let i = 0; i < na; i++) S('circle', { class: 'lrpb-node', cx: 130, cy: ys(i, na), r: 5 }, svg);
        for (let j = 0; j < nb; j++) S('circle', { class: 'lrpb-node', cx: 290, cy: ys(j, nb), r: 5 }, svg);
        T(svg, 118, 100, `${a} values`, 'lrpb-tick', 'end');
        if (p.groups) p.groups.forEach((gname, k) => T(svg, 302, ys(k * 4 + 1.5, nb) + 4, `${gname}, ${b / 3}`, 'lrpb-tick'));
        else T(svg, 302, 100, `${b} values`, 'lrpb-tick');
        return 185;
      }
      if (p.viz === 'window') {
        const n = d.grid, k = 3, s = 86, gap = 22, url = (v, m) => signedMap(v, m);
        const mark = square(svg, 0, 10, s, img, n, 'the image');
        mark(r - k, c - k, 7, 7); mark(r, c);
        const markIn = square(svg, s + gap, 10, s, url(d.dw_map_in, n), n, `channel ${d.dw_channel + 1} before`);
        markIn(r - k, c - k, 7, 7);
        square(svg, 2 * (s + gap), 10, s, url(d.dw_kernel, 7), 7, 'its 7 × 7 weights');
        const markOut = square(svg, 3 * (s + gap), 10, s, url(d.dw_map_out, n), n, 'after');
        markOut(r, c);
        T(svg, 2 * s + gap + gap / 2, 10 + s / 2 + 5, '×', 'lrpb-op', 'middle');
        T(svg, 3 * s + 2 * gap + gap / 2, 10 + s / 2 + 5, '→', 'lrpb-op', 'middle');
        return 130;
      }
      if (p.viz === 'attention') {
        const { w, n } = weightsOf(d), pre = n - G * G, s = 200, x = 0, y = 6;
        const q = state.query, row = w.subarray((pre + q) * n, (pre + q + 1) * n), m = Math.max(...row.subarray(pre));
        S('image', { href: img, x, y, width: s, height: s, preserveAspectRatio: 'none' }, svg);
        for (let t = 0; t < G * G; t++) {
          const v = row[pre + t] / m;
          if (v > 0.02) S('rect', { class: 'lrpb-heat', x: x + (t % G) * s / G, y: y + ((t / G) | 0) * s / G, width: s / G, height: s / G, 'fill-opacity': (0.15 + 0.75 * v).toFixed(2) }, svg);
        }
        S('rect', { class: 'lrpb-frame', x, y, width: s, height: s }, svg);
        S('rect', { class: 'lrpb-query', x: x + (q % G) * s / G, y: y + ((q / G) | 0) * s / G, width: s / G, height: s / G }, svg);
        const hit = S('rect', { x, y, width: s, height: s, fill: 'transparent', style: 'cursor: crosshair' }, svg);
        const pick = (e) => { const b = hit.getBoundingClientRect(), cc = Math.floor((e.clientX - b.left) / b.width * G), rr = Math.floor((e.clientY - b.top) / b.height * G); const nq = Math.max(0, Math.min(G - 1, rr)) * G + Math.max(0, Math.min(G - 1, cc)); if (nq !== state.query) { state.query = nq; render(); } };
        hit.addEventListener('pointermove', pick);
        hit.addEventListener('pointerdown', pick);
        const self = row[pre + q], cls = pre ? row[0] : 0;
        T(svg, s + 24, y + 30, 'the outlined token gives', 'lrpb-tick');
        T(svg, s + 24, y + 52, `${pct(self)} to itself`, 'lrpb-big');
        T(svg, s + 24, y + 80, `${pct(cls)} to the class token,`, 'lrpb-big');
        T(svg, s + 24, y + 98, 'which has no place in the image', 'lrpb-tick');
        T(svg, s + 24, y + 126, `and ${pct(1 - self - cls)} to the other tokens`, 'lrpb-big');
        return s + 14;
      }
      if (p.viz === 'add') {
        const idx = p.b.map((v, i) => i).sort((i, j) => Math.abs(p.b[j]) - Math.abs(p.b[i])).slice(0, 16).sort((i, j) => i - j);
        const vals = idx.flatMap((i) => [p.a[i], p.a[i] + p.b[i], 0]), lo = Math.min(...vals), hi = Math.max(...vals);
        const X = (v) => 70 + (v - lo) / (hi - lo) * 330, rowH = 11;
        S('line', { class: 'lrpb-axis', x1: X(0), x2: X(0), y1: 22, y2: 24 + idx.length * rowH }, svg);
        T(svg, 0, 12, 'channel', 'lrpb-tick');
        S('rect', { class: 'lrpb-bar', x: 150, y: 3, width: 12, height: 8 }, svg); T(svg, 166, 11, 'input', 'lrpb-tick');
        S('rect', { class: 'lrpb-delta', x: 220, y: 3, width: 12, height: 8 }, svg); T(svg, 236, 11, 'what the block adds', 'lrpb-tick');
        idx.forEach((i, k) => {
          const yy = 24 + k * rowH, a = p.a[i], b = p.b[i];
          T(svg, 40, yy + 8, String(i + 1), 'lrpb-tick', 'end');
          S('rect', { class: 'lrpb-bar', x: Math.min(X(0), X(a)), y: yy + 1, width: Math.abs(X(a) - X(0)), height: rowH - 3 }, svg);
          S('rect', { class: 'lrpb-delta', x: Math.min(X(a), X(a + b)), y: yy + 3, width: Math.abs(X(a + b) - X(a)), height: rowH - 7 }, svg);
        });
        T(svg, X(0), 40 + idx.length * rowH, '0', 'lrpb-tick', 'middle');
        return 46 + idx.length * rowH;
      }
      return 0;
    };

    const render = () => {
      const { p } = parts.get(`${state.model}/${state.part}`);
      panel.replaceChildren();
      const h = document.createElement('p'); h.className = 'lrpb-head';
      h.innerHTML = `<b>${p.label}</b><span>${BLOCKS[p.model](p.d).title}</span>`;
      const text = document.createElement('p'); text.textContent = p.text;
      const svg = S('svg', { viewBox: '0 0 420 10', class: 'lrpb-viz' });
      const height = viz(svg, p);
      svg.setAttribute('viewBox', `0 0 420 ${height}`);
      const status = document.createElement('p'); status.className = `lrpb-status${p.rule ? '' : ' lrpb-status--missing'}`; status.textContent = p.status;
      panel.append(h, text, svg, status);
    };

    root.querySelectorAll('.lrpb-diagram').forEach((host) => diagram(host, host.dataset.model));
    select('convnext', 'norm');
  };

  const init = () => document.querySelectorAll('[data-lrp-blocks]').forEach((el) => {
    if (el.dataset.ready) return;
    el.dataset.ready = 'true';
    figure(el);
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

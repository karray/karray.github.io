/* Dense-readout figures of the MIL post. Precomputed readouts (assets/data/gap-mil/<name>.json, written by
   gap_mil_revision/scripts/blog_readouts.py). Each readout ("step") holds the image logits and the 49 cell logits for a
   subset of classes, and every cell's top 5. Attack and layer files hold one readout per attack step or per block.
   The image picker is the figure-gallery dock; this script watches its selection. No model runs in the browser.
   Cells can be picked only in the main readout figure; elsewhere hovering (or tapping) a cell only shows its top 5. */
(() => {
  'use strict';
  if (window.GapMil) return;

  const PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#b07aa1', '#edc948', '#76b7b2', '#ff9da7', '#9c755f', '#bab0ac'];
  const OTHER = '#8c8c8c', TARGET = '#e15759';
  const OPACITY = 55;
  const calm = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* matplotlib's seismic colormap, as control points on [-1, 1]: dark blue, blue, white, red, dark red */
  const SEISMIC = [[0, 0, 77], [0, 0, 255], [255, 255, 255], [255, 0, 0], [128, 0, 0]];
  const cmap = (x) => {
    const t = (Math.max(-1, Math.min(1, x)) + 1) / 2 * (SEISMIC.length - 1);
    const i = Math.min(SEISMIC.length - 2, Math.floor(t)), f = t - i, [a, b] = [SEISMIC[i], SEISMIC[i + 1]];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
  };

  /* ---------- helpers ---------- */
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
  const fmt = (v, d = 2) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`;
  const sum = (a) => a.reduce((s, x) => s + x, 0);
  const range = (n) => [...Array(n).keys()];
  const where = (d, i) => `row ${Math.floor(i / d.grid) + 1}, column ${(i % d.grid) + 1}`;
  const absMax = (arrays) => Math.max(1e-9, ...arrays.flat().map(Math.abs));

  /* ---------- data ---------- */
  const cache = new Map();
  const load = (base, name) => {
    if (!cache.has(name)) cache.set(name, fetch(`${base}${name}.json`).then((r) => r.json()).then(prep));
    return cache.get(name);
  };
  const prep = (d) => {
    d.n = d.grid * d.grid;
    d.name = (c) => d.names[c] ?? String(c);
    for (const s of d.steps) {
      // where the identity holds (maxErr is stored), image scores are shown as the average of the stored cell scores,
      // so rounding never makes the two disagree
      if (d.maxErr != null) s.logit = s.Z.map((row) => sum(row) / d.n);
      s.k = new Map(s.ids.map((c, k) => [c, k]));
      s.top1 = s.top.map((cell) => cell[0][0]);
      s.scores = (c) => (s.k.has(c) ? s.Z[s.k.get(c)] : null);            // the 49 cell logits of class c
      s.image = (c) => (s.k.has(c) ? s.logit[s.k.get(c)] : NaN);          // the image logit of class c
      s.rank = (n) => range(s.ids.length).sort((a, b) => s.logit[b] - s.logit[a]).slice(0, n).map((k) => s.ids[k]);
    }
    d.colour = colours([d]);
    return d;
  };
  // one colour per class, ranked by the number of cells it wins over all readouts of the given files; the target is red
  const colours = (ds) => {
    const wins = new Map();
    for (const d of ds) for (const s of d.steps) for (const c of s.top1) wins.set(c, (wins.get(c) || 0) + 1);
    const t = ds[0].target;
    const map = new Map(t != null ? [[t, TARGET]] : []);
    let p = 0;
    for (const [c] of [...wins].sort((a, b) => b[1] - a[1])) if (!map.has(c)) map.set(c, p < PALETTE.length ? PALETTE[p++] : OTHER);
    return (c) => map.get(c) || OTHER;
  };

  /* ---------- building blocks ---------- */
  // the image with a grid of cells; cells are buttons only where they can be picked
  const stage = (d, { interactive = false, label = '' } = {}) => {
    const cells = range(d.n).map((i) => (interactive
      ? h('button', { type: 'button', class: 'gm-cell', 'aria-label': `${label}cell ${where(d, i)}` }, h('span', { class: 'gm-lab' }))
      : h('span', { class: 'gm-cell is-static' }, h('span', { class: 'gm-lab' }))));
    const el = h('div', { class: `gm-stage${d.grid > 10 ? ' is-fine' : ''}` }, h('img', { class: 'gm-img', src: d.image, alt: '', decoding: 'async' }),   // fine grids: colours only
      h('div', { class: 'gm-grid', style: `--gm-n:${d.grid}` }, cells));
    const paint = (cell, colour, text, title) => {
      cell.style.setProperty('--c', colour);
      cell.firstChild.textContent = text; cell.title = title;
    };
    return {
      el, cells,
      names(s) { cells.forEach((cell, i) => paint(cell, d.colour(s.top1[i]), d.name(s.top1[i]), `${d.name(s.top1[i])} ${fmt(s.top[i][0][1], 1)}`)); },
      values(vals, max = absMax([vals])) { cells.forEach((cell, i) => paint(cell, cmap(vals[i] / max), fmt(vals[i], 1), fmt(vals[i], 2))); },
      mark(set, cls) { cells.forEach((cell, i) => cell.classList.toggle(cls, set.has(i))); },
    };
  };

  const opacity = (...stages) => {
    const input = h('input', { type: 'range', min: 0, max: 100, step: 5, value: OPACITY, 'aria-label': 'Opacity of the readout' });
    const out = h('output', { text: `${OPACITY}%` });
    const set = () => { stages.forEach((st) => st.el.style.setProperty('--gm-alpha', input.value / 100)); out.textContent = `${input.value}%`; };
    input.addEventListener('input', set); set();
    return h('label', { class: 'gm-opacity' }, h('span', { text: 'Opacity' }), input, out);
  };

  // hover previews a cell (mouse), click or tap toggles it
  const bindCells = (st, { hover, toggle }) => {
    st.cells.forEach((cell, i) => {
      cell.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') hover(i); });
      cell.addEventListener('click', () => toggle(i));
    });
    st.el.addEventListener('pointerleave', () => hover(-1));
  };

  // hover (or tap, on touch screens) shows a cell without picking it
  const bindHover = (st, hover) => {
    st.cells.forEach((cell, i) => {
      cell.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') hover(i); });
      cell.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') hover(i); });
    });
    st.el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hover(-1); });
  };

  // ranked classes with bars; a click picks a class
  const list = (d, rows, { pick, active, digits = 2, pickable = () => true }) => {
    const max = Math.max(...rows.map((r) => Math.abs(r.v)), 1e-9);
    return h('ol', { class: 'gm-list' }, rows.map((r) => h('li', {}, h('button', {
      type: 'button', class: `gm-row${active === r.c ? ' is-on' : ''}`, disabled: !pickable(r.c),
      'aria-pressed': String(active === r.c), onclick: () => pick(r.c),
    },
    h('i', { class: 'gm-sw', style: `background:${d.colour(r.c)}` }),
    h('span', { class: 'gm-name', text: d.name(r.c) }),
    h('span', { class: 'gm-val', text: r.n != null ? `×${r.n}` : fmt(r.v, digits) }),
    h('span', { class: 'gm-track' }, h('span', { class: `gm-fill${r.v < 0 ? ' is-neg' : ''}`, style: `width:${Math.abs(r.v) / max * 100}%;background:${d.colour(r.c)}` }))))));
  };
  const card = (title, ...kids) => h('section', { class: 'gm-card' }, title ? h('h4', { class: 'gm-h', text: title }) : null, ...kids);
  const note = (text) => h('p', { class: 'gm-note', text });
  const sub = (text) => h('p', { class: 'gm-sub', text });

  // image-level classes of one readout; extra classes are appended when missing (start and target of an attack)
  const imageRows = (s, n, extra = []) => {
    const cs = s.rank(n);
    for (const c of extra) if (c != null && !cs.includes(c) && s.k.has(c)) cs.push(c);
    return cs.map((c) => ({ c, v: s.image(c) }));
  };
  // what the picked (or hovered) cells say: the union of their top 5, at the highest logit among them
  const cellRows = (s, cells) => {
    const best = new Map();
    for (const i of cells) for (const [c, v] of s.top[i]) best.set(c, Math.max(best.get(c) ?? -Infinity, v));
    return [...best].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([c, v]) => ({ c, v }));
  };
  // the classes on the grid with the number of cells each one wins
  const gridRows = (s) => {
    const n = new Map();
    for (const c of s.top1) n.set(c, (n.get(c) || 0) + 1);
    return [...n].sort((a, b) => b[1] - a[1]).map(([c, k]) => ({ c, v: k, n: k }));
  };

  // what the numbers mean; with a class picked, the same card states the average for that class
  const help = (d, s, c) => card('Reading the scores',
    note('Every number is a logit, the score the classifier gives a class before softmax turns scores into probabilities. Higher means more of that class.'),
    c == null
      ? note(`Each cell is scored by the same classifier as if it were an image of its own. The image score is the average of the ${d.n} cell scores.`)
      : h('p', { class: 'gm-eq' }, `The numbers on the image are the ${d.name(c)} scores of the ${d.n} cells. Their average, `,
        h('strong', { text: fmt(sum(s.scores(c)) / d.n) }), `, is exactly the model’s image score for ${d.name(c)}.`),
    c == null ? note('Pick a class to see its score in every cell. Tap cells to compare what they see.') : null);

  // steps (attack steps or blocks) as small grids that work as a slider: click, drag, arrow keys. One row per file.
  const filmstrip = (ds, set, { label, text }) => {
    const T = ds[0].steps.length - 1, target = ds[0].target;
    // with two rows (before and after) the step label sits between them and separates them; with one row it sits below
    const grid = (d, t) => h('span', { class: 'gm-mini-grid', style: `--gm-n:${d.grid}` }, d.steps[t].top1.map((c) => h('i', {
      style: `background:${c === target ? TARGET : c === d.start ? d.colour(d.start) : 'var(--fig-track)'}`,
    })));
    const minis = range(T + 1).map((t) => {
      const lab = h('small', { text: label(t) }), maps = ds.map((d) => grid(d, t));
      return h('span', { class: 'gm-mini' }, ds.length === 2 ? [maps[0], lab, maps[1]] : [...maps, lab]);
    });
    // a gel frame: one rounded rectangle whose two edges move on springs. While dragging, the edge on the pointer's side
    // follows the pointer and the other edge sticks to the current step; past halfway to the next step the step changes
    // and the stuck edge lets go and springs over. The stretch is capped, so the frame never splits or runs away.
    const gel = h('span', { class: 'gm-gel', 'aria-hidden': 'true' });
    const el = h('div', { class: 'gm-film', role: 'slider', tabindex: 0, 'aria-valuemin': 0, 'aria-valuemax': T }, gel, minis);
    let cur = -1, drag = false, pointer = null;
    const edge = { L: { x: 0, v: 0 }, R: { x: 0, v: 0 } };
    let frame = 0, placed = false;
    const box = (t) => [minis[t].offsetLeft, minis[t].offsetLeft + minis[t].offsetWidth];
    const targets = () => {
      const w = minis[0].offsetWidth, [l, r] = box(Math.max(0, cur));
      if (pointer == null) return [l, r, null];
      const p = Math.max(box(0)[0] + w / 2, Math.min(box(T)[1] - w / 2, pointer));
      return p >= (l + r) / 2
        ? [Math.max(l, p + w / 2 - 2 * w), p + w / 2, 'R']                 // pulled right: the left edge sticks (at most 2 cells long)
        : [p - w / 2, Math.min(r, p - w / 2 + 2 * w), 'L'];                 // pulled left: the right edge sticks
    };
    const draw = () => {
      const w = minis[0].offsetWidth, len = edge.R.x - edge.L.x, stretch = Math.max(0, Math.min(1, (len - w) / w));
      gel.style.transform = `translateX(${edge.L.x}px) scaleY(${1 - 0.12 * stretch})`;
      gel.style.width = `${len}px`;
    };
    const tick = () => {
      const [tl, tr, held] = targets();
      const moving = (edge.R.x + edge.L.x) / 2 < (tl + tr) / 2 ? 'R' : 'L';   // the leading edge is quicker than the trailing one
      let busy = false;
      for (const [k, t] of [['L', tl], ['R', tr]]) {
        const e = edge[k];
        if (k === held) { e.x = t; e.v = 0; continue; }                     // the edge held by the pointer follows it exactly
        const stiff = k === moving ? 0.3 : 0.17;
        e.v = (e.v + (t - e.x) * stiff) * 0.62;
        e.x = Math.max(box(0)[0], Math.min(box(T)[1], e.x + e.v));            // never past the ends of the strip
        if (Math.abs(t - e.x) > 0.3 || Math.abs(e.v) > 0.3) busy = true; else { e.x = t; e.v = 0; }
      }
      draw();
      frame = busy || drag ? requestAnimationFrame(tick) : 0;
    };
    const kick = () => {
      if (!placed || calm.matches) {                                          // first placement, or no motion wanted: jump
        [edge.L.x, edge.R.x] = targets(); edge.L.v = edge.R.v = 0; placed = true; draw(); return;
      }
      if (!frame) frame = requestAnimationFrame(tick);
    };
    const go = (t) => { if (t !== cur) set(t); };
    const follow = (e) => {
      const r = el.getBoundingClientRect();
      pointer = e.clientX - r.left;
      let best = 0;
      minis.forEach((m, t) => { const c = m.offsetLeft + m.offsetWidth / 2; if (Math.abs(c - pointer) < Math.abs(minis[best].offsetLeft + minis[best].offsetWidth / 2 - pointer)) best = t; });
      go(best);
      kick();
    };
    el.addEventListener('pointerdown', (e) => { drag = true; el.setPointerCapture(e.pointerId); follow(e); });
    el.addEventListener('pointermove', (e) => { if (drag) follow(e); });
    const end = () => { if (drag) { drag = false; pointer = null; kick(); } };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('keydown', (e) => {
      const next = { ArrowLeft: cur - 1, ArrowDown: cur - 1, ArrowRight: cur + 1, ArrowUp: cur + 1, Home: 0, End: T }[e.key];
      if (next == null) return;
      e.preventDefault(); go(Math.max(0, Math.min(T, next)));
    });
    new ResizeObserver(() => { placed = false; kick(); }).observe(el);
    return {
      el: h('div', { class: 'gm-filmwrap' }, el, note(text)),
      update(t, valuetext) {
        cur = t;
        requestAnimationFrame(kick);
        minis.forEach((m, j) => m.classList.toggle('is-on', j === t));
        el.setAttribute('aria-valuenow', t);
        el.setAttribute('aria-valuetext', valuetext);
      },
    };
  };

  /* ---------- the main figure: one image, cells can be picked ---------- */
  const readout = (app, d) => {
    const s = d.steps[0];
    const st = stage(d, { interactive: true });
    const state = { cls: null, picked: new Set(), hover: -1 };
    const left = h('div', { class: 'gm-left' }), right = h('div', { class: 'gm-right' });
    const pickable = (c) => !!s.scores(c);
    // picking a class leaves cell mode, picking a cell leaves class mode
    const pick = (c) => { state.picked.clear(); state.hover = -1; state.cls = state.cls === c ? null : c; render(); };
    const render = () => {
      if (state.cls != null) st.values(s.scores(state.cls)); else st.names(s);
      st.mark(state.picked, 'is-picked'); st.mark(new Set(state.hover >= 0 ? [state.hover] : []), 'is-hover');
      left.replaceChildren(
        card('Image', note('The model’s answer. Pick a class to see its score in every cell.'), list(d, imageRows(s, 5), { pick, active: state.cls, pickable })),
        help(d, s, state.cls));
      const look = state.hover >= 0 ? [state.hover] : [...state.picked];
      right.replaceChildren(look.length
        ? card(look.length === 1 ? `Cell, ${where(d, look[0])}` : `${look.length} cells`,
          note(look.length === 1 ? 'Its top 5, scored by the same classifier.' : 'Their top classes, at the highest score among them.'),
          list(d, cellRows(s, look), { pick, active: state.cls, digits: 1, pickable }),
          state.picked.size ? h('button', { type: 'button', class: 'gm-link', text: 'Clear the cells', onclick: () => { state.picked.clear(); render(); } }) : null)
        : card('Cells', note('Each cell’s top class, and how many cells it wins. Hover or tap cells for their own top 5.'),
          list(d, gridRows(s), { pick, active: state.cls, pickable })));
    };
    bindCells(st, {
      hover: (i) => { if (state.cls == null && state.hover !== i) { state.hover = i; render(); } },   // no hover while a class is shown
      toggle: (i) => {
        state.hover = -1;
        if (state.cls != null) state.cls = null;
        state.picked.has(i) ? state.picked.delete(i) : state.picked.add(i);
        render();
      },
    });
    app.replaceChildren(h('div', { class: 'gm-3col' }, h('div', { class: 'gm-centre' }, st.el, opacity(st)), left, right));
    render();
  };

  /* ---------- an attack: one image over the steps ---------- */
  const attack = (app, d) => {
    const st = stage(d);
    const state = { t: d.steps.length - 1, cls: null, hover: -1 };
    const left = h('div', { class: 'gm-left' }), right = h('div', { class: 'gm-right' });
    const pick = (c) => { state.cls = state.cls === c ? null : c; render(); };
    const T = d.steps.length - 1;
    const flipAt = d.steps.findIndex((x) => x.rank(1)[0] === d.target);
    const film = filmstrip([d], (t) => { state.t = t; render(); }, {
      label: (t) => (t === 0 || t === T || t === flipAt ? String(t) : ''),
      text: `Each small grid is one attack step. Red cells vote ${d.name(d.target)}. Drag along the strip to move through the attack.`,
    });
    const render = () => {
      const s = d.steps[state.t];
      if (state.cls != null && s.scores(state.cls)) st.values(s.scores(state.cls), absMax(d.steps.map((x) => x.scores(state.cls) || []))); else st.names(s);
      const pickable = (c) => !!s.scores(c);
      left.replaceChildren(card(`Image, step ${state.t}`, note(state.t >= flipAt ? 'The prediction has flipped.' : 'The prediction has not flipped yet.'),
        list(d, imageRows(s, 5, [d.start, d.target]), { pick, active: state.cls, pickable })));
      st.mark(new Set(state.hover >= 0 ? [state.hover] : []), 'is-hover');
      right.replaceChildren(state.hover >= 0
        ? card(`Cell, ${where(d, state.hover)}`, note('Its top 5 at this step.'), list(d, s.top[state.hover].map(([c, v]) => ({ c, v })), { pick, active: state.cls, digits: 1, pickable }))
        : card('Cells', note('Each cell’s top class, and how many cells it wins. Hover or tap a cell for its top 5.'), list(d, gridRows(s), { pick, active: state.cls, pickable })));
      film.update(state.t, `Step ${state.t} of ${T}`);
    };
    bindHover(st, (i) => { state.hover = i; render(); });
    // the strip above, the lists beside the image and exactly as tall as it
    app.replaceChildren(h('div', { class: 'gm-att' }, h('div', { class: 'gm-strip' }, film.el), left, st.el, right, h('div', { class: 'gm-op' }, opacity(st))));
    render();
  };

  /* ---------- two images side by side (an edit), optionally over several blocks ---------- */
  // image: show the image-level answer (single readouts only), as a compact top 3 under each picture; the side panels
  // then list only the cells and span exactly the image height
  const pair = (app, ds, labels, { image = true } = {}) => {
    const [a, b] = ds;
    a.colour = b.colour = colours(ds);
    const T = a.steps.length - 1;
    const key = a.target ?? a.steps[T].rank(1)[0];          // the original's answer stays in both image lists
    const sts = ds.map((d) => stage(d));
    const state = { t: T, cls: null, hover: -1 };
    const sides = [h('div', { class: 'gm-side' }), h('div', { class: 'gm-side' })];
    const under = T === 0 && image;
    const unders = [h('div', { class: 'gm-under gm-ua' }), h('div', { class: 'gm-under gm-ub' })];
    const pick = (c) => { state.cls = state.cls === c ? null : c; render(); };
    const film = T > 0 ? filmstrip(ds, (t) => { state.t = t; render(); }, {
      label: (t) => a.labels[t].replace(/stage (\d+), block (\d+)/, '$1.$2').replace(/^block /, ''),
      text: `Each column is one block of the network, from early to last${/stage/.test(a.labels[0]) ? ' (stage.block)' : ''}: top row ${labels[0].toLowerCase()}, bottom row ${labels[1].toLowerCase()}. Red cells vote ${a.name(a.target)}.`,
    }) : null;
    const render = () => {
      const steps = ds.map((d) => d.steps[state.t]);
      const max = state.cls != null ? absMax(ds.flatMap((d) => d.steps.map((s) => s.scores(state.cls) || []))) : 1;
      ds.forEach((d, j) => {
        const s = steps[j], pickable = (c) => !!s.scores(c);
        if (state.cls != null && s.scores(state.cls)) sts[j].values(s.scores(state.cls), max); else sts[j].names(s);
        sts[j].mark(new Set(state.hover >= 0 ? [state.hover] : []), 'is-hover');
        // the cells at the side (a hovered cell, or the counts); the image-level answer under the picture
        const at = T > 0 ? `, ${a.labels[state.t]}` : '';
        if (under) unders[j].replaceChildren(h('p', { class: 'gm-h', text: 'Image' }), list(d, imageRows(s, 3, [key]), { pick, active: state.cls, pickable }));
        sides[j].replaceChildren(
          card(state.hover >= 0 ? `Cell, ${where(d, state.hover)}${at}` : `Cells${at}`, h('div', { class: 'gm-swap' }, state.hover >= 0   // fixed height, so hovering does not move the page
            ? list(d, s.top[state.hover].map(([c, v]) => ({ c, v })), { pick, active: state.cls, digits: 1, pickable })
            : list(d, gridRows(s), { pick, active: state.cls, pickable }))));
      });
      if (film) film.update(state.t, a.labels[state.t]);
    };
    sts.forEach((st) => bindHover(st, (i) => { state.hover = i; render(); }));
    // captions in their own row, so each side panel spans exactly its image
    sides[0].classList.add('gm-sa'); sides[1].classList.add('gm-sb');
    sts[0].el.classList.add('gm-ia'); sts[1].el.classList.add('gm-ib');
    app.replaceChildren(h('div', { class: `gm-pair${film ? ' has-film' : ''}${under ? ' has-under' : ''}` },
      film ? h('div', { class: 'gm-strip' }, film.el) : null,
      h('p', { class: 'gm-cap gm-ca', text: labels[0] }), h('p', { class: 'gm-cap gm-cb', text: labels[1] }),
      sides[0], sts[0].el, sts[1].el, sides[1], ...(under ? unders : []),
      h('div', { class: 'gm-op' }, opacity(...sts))));
    render();
  };

  /* ---------- wiring ---------- */
  // the model menu: data-models="key:Label;key:Label", the first is the default
  const MODELS = 'resnet50:ResNet-50;convnext:ConvNeXt-B;swin:Swin-B;maxvit:MaxViT-B';

  const mount = (root) => {
    const app = root.querySelector('[data-gm-app]');
    const kind = root.dataset.gm || 'readout';
    const base = root.dataset.base;
    const thumbs = [...root.querySelectorAll('.fg-thumb')];
    const models = (root.dataset.models || MODELS).split(';').map((m) => m.split(':'));
    const state = { model: models[0][0], item: null };
    const show = async () => {
      const file = (n) => load(base, `${state.model}/${n}`);
      try {
        app.classList.add('is-loading');
        if (kind === 'pair') pair(app, await Promise.all(state.item.split('|').map(file)), (root.dataset.labels || 'Before|After').split('|'), { image: root.dataset.image !== 'off' });
        else if (kind === 'attack') attack(app, await file(state.item));
        else readout(app, await file(state.item));
      } catch (err) {
        app.replaceChildren(h('p', { class: 'fg-note', text: 'This figure could not load its data.' }));
        console.error(err);
      } finally { app.classList.remove('is-loading'); }
    };
    const select = h('select', { 'aria-label': 'Model', onchange: () => { state.model = select.value; show(); } },
      models.map(([k, label]) => h('option', { value: k, text: label })));
    app.before(h('label', { class: 'gm-model' }, h('span', { text: 'Model' }), select));
    if (thumbs.length) {
      new MutationObserver(() => {
        const on = thumbs.find((t) => t.getAttribute('aria-selected') === 'true');
        if (on && on.dataset.item !== state.item) { state.item = on.dataset.item; show(); }
      }).observe(root.querySelector('.fg-dock'), { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
      state.item = (thumbs.find((t) => t.dataset.selected === 'true') || thumbs[0]).dataset.item;
    } else state.item = root.dataset.item;
    show();
  };

  const init = () => document.querySelectorAll('figure[data-gm]').forEach((root) => {
    if (root.dataset.gmReady) return;
    root.dataset.gmReady = 'true';
    mount(root);
  });
  window.GapMil = { init };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

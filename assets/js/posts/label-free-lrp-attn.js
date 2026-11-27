(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const SUB = '₁₂₃';
  const INV_SQRT2 = 1 / Math.SQRT2;
  const YS = [294, 418, 542];
  const CX = 670;
  const V0 = [[2.4, -0.4], [1.7, -0.5], [2.3, -2.8]];

  const LAYOUT = {
    wide: {
      tall: false,
      vb: '0 0 1364 636',
      sCols: [248, 282, 316],
      fCols: [418, 464, 510],
      bCols: [824, 868, 912],
      softBusY: [94, 124, 154],
      softStartX: 214,
      softLabelX: 218,
      softArrowX: [333, 375],
      softPrimeX: 380,
      softSumX: 548,
      softRowY0: 98,
      dropPillY: 176,
      dropTextY: 190,
      bwdRowX: 778,
      bwdRowTxtX: 784,
      bwdRowSumX: 949,
      bwdRowY0: 95,
      bwdCardX: 984,
      bwdBusX: 966,
      bwdBusMidY: 124,
      cancelX: 786,
      cancelY: 174,
      midW: 174,
      rightW: 200,
      centerTot: [CX, 624],
      rightTot: [1254, 624]
    },
    tall: {
      tall: true,
      vb: '0 0 624 1142',
      xs: [108, 312, 516],
      sCols: [172, 206, 240],
      fCols: [342, 388, 434],
      bCols: [184, 228, 272],
      softBusY: [682, 712, 742],
      softStartX: 138,
      softLabelX: 142,
      softArrowX: [257, 299],
      softPrimeX: 304,
      softSumX: 472,
      softRowY0: 188,
      dropPillY: 266,
      dropTextY: 280,
      bwdRowX: 138,
      bwdRowTxtX: 144,
      bwdRowSumX: 309,
      bwdRowY0: 683,
      bwdCardX: 344,
      bwdBusX: 326,
      bwdBusMidY: 712,
      cancelX: 146,
      cancelY: 762,
      midW: 184,
      rightW: 192,
      centerTot: [516, 610],
      rightTot: [516, 1134]
    }
  };

  const svg = (tag, attrs = {}, ...kids) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    node.append(...kids.flat(Infinity).filter((c) => c != null && c !== false));
    return node;
  };

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  const safeDen = (z, eps = 1e-9) => (Math.abs(z) > eps ? z : (z >= 0 ? eps : -eps));
  const fmt = (v, d = 2) => {
    if (!Number.isFinite(v)) return '∞';
    const s = v.toFixed(d);
    return (Number(s) === 0 ? (0).toFixed(d) : s).replace('-', '−');
  };
  const fmtS = (v, d = 2) => (v > 0.005 ? '+' : '') + fmt(v, d);
  const fmtVec = (v, d = 2) => `[${fmt(v[0], d)}, ${fmt(v[1], d)}]`;
  const fmtSumParts = (a, b, res) => [`${fmt(a)} ${b < 0 ? '−' : '+'} ${fmt(Math.abs(b))} = `, fmtS(res)];

  const softmax = (s) => {
    const mx = Math.max(...s);
    const e = s.map((v) => Math.exp(v - mx));
    const z = sum(e);
    return e.map((v) => v / z);
  };

  const softmaxZPlus = (s, a, rel) => {
    const pos = s.map((v) => Math.max(0, v));
    const neg = s.map((v) => Math.max(0, -v));
    const sa = sum(a.map((ai, i) => ai * s[i]));
    const spread = sum(a.map((ai, i) => ai * neg[i]));
    const ratio = a.map((ai, i) => {
      const bias = ai * (1 - s[i] + sa);
      const denom = ai * ((1 - ai) * pos[i] + spread - ai * neg[i]) + Math.max(0, bias);
      return rel[i] / (denom + 1e-12);
    });
    const total = sum(a.map((ai, i) => ai * ratio[i]));
    return a.map((ai, i) => ai * (1 - ai) * pos[i] * ratio[i] + ai * neg[i] * (total - ai * ratio[i]));
  };

  const attnRow = (v, s, R, mode) => {
    const a = softmax(s);
    const y = sum(a.map((aj, j) => aj * v[j]));
    const n = v.length;
    let rv = new Array(n).fill(0);
    let rs = new Array(n).fill(0);
    if (mode === 'cp') {
      const c = R / safeDen(y);
      rv = a.map((aj, j) => aj * v[j] * c);
    } else if (mode === 'grad') {
      rv = a.map((aj) => aj * R);
      rs = a.map((aj, j) => aj * (v[j] - y) * R);
    } else {
      const c = R / safeDen(y);
      rv = a.map((aj, j) => 0.5 * aj * v[j] * c);
      const rA = a.map((aj, j) => 0.5 * aj * v[j] * c);
      rs = softmaxZPlus(s, a, rA);
    }
    return { a, y, rv, rs };
  };

  const bezCurve = (a, b, sag) => {
    const k = sag ?? Math.max(40, Math.abs(b[0] - a[0]) * 0.44);
    const dir = Math.sign(b[0] - a[0]) || 1;
    return [a, [a[0] + dir * k, a[1]], [b[0] - dir * k, b[1]], b];
  };

  const bezCurveVert = (a, b, sag) => {
    const k = sag ?? Math.max(40, Math.abs(b[1] - a[1]) * 0.44);
    const dir = Math.sign(b[1] - a[1]) || 1;
    return [a, [a[0], a[1] + dir * k], [b[0], b[1] - dir * k], b];
  };

  const pathD = (p) => `M${p[0]}C${p[1]} ${p[2]} ${p[3]}`;

  const bez = (p, t) => [0, 1].map((k) =>
    (1 - t) ** 3 * p[0][k] + 3 * (1 - t) ** 2 * t * p[1][k] + 3 * (1 - t) * t ** 2 * p[2][k] + t ** 3 * p[3][k]);

  const bezAt = (p, target, axis) => {
    let lo = 0, hi = 1;
    const fwd = p[3][axis] >= p[0][axis];
    for (let it = 0; it < 16; it++) {
      const mid = 0.5 * (lo + hi);
      const val = bez(p, mid)[axis];
      if (fwd ? val < target : val > target) lo = mid;
      else hi = mid;
    }
    return bez(p, 0.5 * (lo + hi));
  };

  const bezAtX = (p, targetX) => bezAt(p, targetX, 0);
  const bezAtY = (p, targetY) => bezAt(p, targetY, 1);

  const addText = (g, x, y, s, cls, anchor = 'middle') =>
    g.appendChild(svg('text', { x, y, class: cls, 'text-anchor': anchor }, s));

  const wire = (L, p, isFwd) => {
    const d = pathD(p);
    if (isFwd) {
      const guide = L.track.appendChild(svg('path', { d, class: 'lrpa-fwd-guide' }));
      const band = L.fwd.appendChild(svg('path', { d, class: 'lrpa-fwd-band' }));
      const flow = L.flow.appendChild(svg('path', { d, class: 'lrpa-fwd-flow' }));
      return (aVal, vecNorm, active) => {
        const gw = 0.8 + aVal * 6.5;
        const cw = Math.min(vecNorm * 11, 22);
        guide.style.strokeWidth = `${gw}px`;
        band.style.strokeWidth = `${Math.max(cw, active ? 1.5 : 0.6)}px`;
        flow.style.strokeWidth = `${Math.min(2.4, cw * 0.34)}px`;
        const dim = !active;
        guide.classList.toggle('is-dim', dim);
        band.classList.toggle('is-dim', dim);
        flow.classList.toggle('is-dim', dim || cw <= 1.8);
      };
    }
    const track = L.track.appendChild(svg('path', { d, class: 'lrpa-track' }));
    const stream = L.stream.appendChild(svg('path', { d, class: 'lrpa-stream' }));
    const flow = L.flow.appendChild(svg('path', { d, class: 'lrpa-bwd-flow' }));
    return (q, active) => {
      const w = Math.min(Math.abs(q) * 14, 22);
      stream.style.strokeWidth = `${w}px`;
      stream.classList.toggle('is-neg', q < -1e-6);
      flow.style.strokeWidth = `${Math.min(2.4, w * 0.34)}px`;
      const dim = !active;
      track.classList.toggle('is-dim', dim);
      stream.classList.toggle('is-dim', dim);
      flow.classList.toggle('is-dim', dim || w <= 1.8);
    };
  };

  const makeChip = (g, { x, y, w = 66, get, set, lim = [-3, 3], step = 0.1, label, onTouch }) => {
    const t = svg('text', { x, y: y + 5, 'text-anchor': 'middle' });
    const arrow = (dx, s) => svg('text', { x: x + dx, y: y + 4.5, 'text-anchor': 'middle', class: 'lrpa-chip-arrow' }, s);
    let x0 = 0, v0 = 0, scale = 1, moved = false;
    const put = (q) => set(Number(clamp(Math.round(q / step) * step, lim[0], lim[1]).toFixed(4)));
    const node = g.appendChild(svg('g', {
      class: 'lrpa-chip', tabindex: 0, role: 'slider', 'aria-label': label,
      'aria-valuemin': lim[0], 'aria-valuemax': lim[1],
      onpointerdown: (e) => {
        node.setPointerCapture(e.pointerId);
        const sc = node.ownerSVGElement;
        x0 = e.clientX; v0 = get(); moved = false;
        scale = sc.viewBox.baseVal.width / sc.getBoundingClientRect().width;
        node.classList.add('is-drag'); onTouch();
      },
      onpointermove: (e) => {
        if (!node.hasPointerCapture(e.pointerId)) return;
        if (Math.abs(e.clientX - x0) >= 4) moved = true;
        if (moved) put(v0 + (e.clientX - x0) * scale / 40);
      },
      onpointerup: (e) => {
        node.classList.remove('is-drag');
        if (!moved && Math.abs(e.clientX - x0) < 4) {
          const r = node.getBoundingClientRect();
          const relX = (e.clientX - r.left) / (r.width || 1);
          if (relX < 0.35) put(get() - step);
          else if (relX > 0.65) put(get() + step);
        }
      },
      onpointercancel: () => node.classList.remove('is-drag'),
      onkeydown: (e) => {
        const k = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10 }[e.key];
        if (k) { e.preventDefault(); put(get() + k * step); onTouch(); }
      }
    }, svg('rect', { x: x - w / 2, y: y - 16, width: w, height: 32, rx: 16 }), arrow(-w / 2 + 11, '‹'), t, arrow(w / 2 - 11, '›')));
    return () => { t.textContent = fmt(get(), 1); node.setAttribute('aria-valuenow', get()); };
  };

  const initFigure = (root) => {
    const ruleBtns = [...root.querySelectorAll('[data-rule]')];
    const tauInp = root.querySelector('[data-attn-tau]');
    const tauOut = root.querySelector('[data-attn-tau-out]');
    const note = root.querySelector('[data-attn-note]');
    const svgEl = root.querySelector('svg.lrp-attn-scene');
    const frameWide = root.querySelector('[data-frame="wide"]');
    const frameTall = root.querySelector('[data-frame="tall"]');
    const bwdSoftBoxWide = root.querySelector('[data-bwd-softbox-wide]');
    const bwdSoftBoxTall = root.querySelector('[data-bwd-softbox-tall]');

    const L = Object.fromEntries(
      ['track', 'fwd', 'stream', 'flow', 'nodes', 'text'].map((n) => [n, root.querySelector(`[data-layer="${n}"]`)])
    );

    const st = { v: V0.map((p) => [...p]), tau: Number(tauInp.value) || 1, mode: 'cp', hover: null };
    const touch = () => root.classList.add('lrp-attn-touched');
    let lastPointer = 'mouse';

    root.addEventListener('pointerdown', (e) => { lastPointer = e.pointerType || 'mouse'; }, { capture: true });
    root.addEventListener('keydown', () => { lastPointer = 'keyboard'; }, { capture: true });
    svgEl.addEventListener('click', (e) => {
      if (!e.target.closest('.lrpa-srow, .lrpa-mid-btn, .lrpa-chip') && st.hover !== null) {
        st.hover = null;
        update();
      }
    });

    const bindRow = (el, i, isPill = false) => {
      const toggleHover = () => { st.hover = st.hover === i ? null : i; update(); };
      el.addEventListener('pointerenter', (e) => {
        if (e.pointerType === 'mouse') { st.hover = i; update(); }
      });
      el.addEventListener('pointerleave', (e) => {
        if (e.pointerType === 'mouse' && st.hover === i) { st.hover = null; update(); }
      });
      el.addEventListener('click', (e) => {
        if (lastPointer !== 'mouse' || e.detail === 0) {
          e.stopPropagation();
          toggleHover();
        }
      });
      if (!isPill) return;
      el.addEventListener('focus', () => { if (lastPointer === 'keyboard') { st.hover = i; update(); } });
      el.addEventListener('blur', () => { if (lastPointer === 'keyboard' && st.hover === i) { st.hover = null; update(); } });
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleHover(); }
      });
    };

    let ui = null;
    let isTall = null;

    const buildScene = (tall) => {
      Object.values(L).forEach((layer) => layer.replaceChildren());
      const g = LAYOUT[tall ? 'tall' : 'wide'];
      svgEl.setAttribute('viewBox', g.vb);
      svgEl.classList.toggle('is-tall', tall);
      frameWide.toggleAttribute('hidden', tall);
      frameTall.toggleAttribute('hidden', !tall);

      const chips = [];
      const softPathsTall = [
        'M108,108 L108,128 Q108,132 102,132 L32,132 Q26,132 26,138 L26,182 Q26,188 32,188 L138,188',
        'M312,108 L312,111 Q312,115 308,115 L26,115 Q20,115 20,121 L20,208 Q20,214 26,214 L138,214',
        'M516,108 L516,117 Q516,121 512,121 L20,121 Q14,121 14,127 L14,234 Q14,240 20,240 L138,240'
      ];
      [0, 1, 2].forEach((j) => {
        const cx = tall ? g.xs[j] : 98;
        const y = tall ? 70 : YS[j];
        const cardX = tall ? cx - 92 : 10;
        const cardW = tall ? 184 : 176;
        L.nodes.appendChild(svg('rect', { x: cardX, y: y - 38, width: cardW, height: 76, rx: 12, class: 'lrpa-token-card' }));
        addText(L.text, cardX + 12, y - 16, `t${SUB[j]}`, 'lrpa-port-name', 'start');
        addText(L.text, cardX + 34, y - 17, '∈ ℝ²', 'lrpa-box-sub', 'start');
        addText(L.text, cx - 79, y + 20, '[', 'lrpa-bracket', 'middle');
        chips.push(makeChip(L.text, {
          x: cx - 39, y: y + 14, get: () => st.v[j][0],
          set: (val) => { st.v[j][0] = val; update(); },
          label: `first coordinate of t${j + 1}`, onTouch: touch
        }));
        addText(L.text, cx - 2, y + 19, ',', 'lrpa-comma', 'middle');
        chips.push(makeChip(L.text, {
          x: cx + 35, y: y + 14, get: () => st.v[j][1],
          set: (val) => { st.v[j][1] = val; update(); },
          label: `second coordinate of t${j + 1}`, onTouch: touch
        }));
        addText(L.text, cx + 75, y + 20, ']', 'lrpa-bracket', 'middle');
        const td = tall
          ? softPathsTall[j]
          : pathD(bezCurve([186, YS[j] - 18], [214, 98 + j * 26], 16));
        L.track.appendChild(svg('path', { d: td, class: 'lrpa-fwd-soft-track' }));
      });

      const softRows = [0, 1, 2].map((i) => {
        const ry = g.softRowY0 + i * 26;
        const rowG = L.nodes.appendChild(svg('g', { class: 'lrpa-srow' }));
        bindRow(rowG, i);
        rowG.appendChild(svg('rect', { x: g.softStartX, y: ry - 12, width: 348, height: 24, class: 'lrpa-srow-bg' }));
        rowG.appendChild(svg('text', { x: g.softLabelX, y: ry + 4, class: 'lrpa-box-sub', 'text-anchor': 'start' }, `t${SUB[i]}`));

        const sCells = g.sCols.map((sx) => ({
          r: rowG.appendChild(svg('rect', { x: sx - 15, y: ry - 10, width: 30, height: 20, rx: 5, class: 'lrpa-score-pill' })),
          t: rowG.appendChild(svg('text', { x: sx, y: ry + 3.5, 'text-anchor': 'middle', class: 'lrpa-score-txt' }, ''))
        }));
        const [ax1, ax2] = g.softArrowX;
        rowG.appendChild(svg('path', {
          d: `M${ax1},${ry} L${ax2},${ry} M${ax2 - 5},${ry - 3.5} L${ax2},${ry} L${ax2 - 5},${ry + 3.5}`, class: 'lrpa-bus'
        }));
        rowG.appendChild(svg('text', { x: g.softPrimeX, y: ry + 4, class: 'lrpa-box-sub', 'text-anchor': 'start' }, `t${SUB[i]}′`));

        const cells = g.fCols.map((cx) => ({
          r: rowG.appendChild(svg('rect', { x: cx - 18, y: ry - 10, width: 36, height: 20, rx: 5, class: 'lrpa-cell-chip' })),
          t: rowG.appendChild(svg('text', { x: cx, y: ry + 4, 'text-anchor': 'middle', class: 'lrpa-cell-val' }, ''))
        }));
        rowG.appendChild(svg('text', { x: g.softSumX, y: ry + 4, 'text-anchor': 'middle', class: 'lrpa-box-sub' }, '1.00'));
        return { rowG, sCells, cells };
      });

      const dropEls = g.fCols.map((cx) => {
        const line = L.flow.appendChild(svg('path', { d: '', class: 'lrpa-drop-line' }));
        const mulG = L.nodes.appendChild(svg('g'));
        mulG.appendChild(svg('circle', { cx: 0, cy: 0, r: 9, class: 'lrpa-mul-node' }));
        mulG.appendChild(svg('text', { x: 0, y: 4, 'text-anchor': 'middle', class: 'lrpa-mul-sign' }, '×'));
        const pill = L.text.appendChild(svg('g', { class: 'lrpa-drop-pill' }));
        pill.appendChild(svg('rect', { x: cx - 22, y: g.dropPillY, width: 44, height: 20, rx: 10 }));
        const pTxt = pill.appendChild(svg('text', { x: cx, y: g.dropTextY, 'text-anchor': 'middle' }, ''));
        return { cx, line, mulG, pTxt };
      });

      const fwdCurves = [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
        (tall
          ? bezCurveVert([g.xs[j] + (i - 1) * 10, 338], [g.xs[i] + (j - 1) * 10, 508], 68)
          : bezCurve([186, YS[j] + (i - 1) * 9], [583, YS[i] + (j - 1) * 9], 145))));
      const fwdWires = fwdCurves.map((row) => row.map((p) => wire(L, p, true)));

      const midTokens = [0, 1, 2].map((i) => {
        const mx = tall ? g.xs[i] : CX;
        const my = tall ? 545 : YS[i];
        const pillG = L.nodes.appendChild(svg('g', {
          class: 'lrpa-mid-btn', tabindex: 0, role: 'button',
          'aria-label': `Toggle highlight for updated token vector t${i + 1}'`
        }));
        bindRow(pillG, i, true);
        pillG.appendChild(svg('rect', { x: mx - g.midW / 2, y: my - 37, width: g.midW, height: 74, rx: 22, class: 'lrpa-mid-pill' }));
        const eq = pillG.appendChild(svg('text', { x: mx, y: my - 18, 'text-anchor': 'middle', class: 'lrpa-mid-eq' }, ''));
        const line2 = pillG.appendChild(svg('text', { x: mx, y: my + 1, 'text-anchor': 'middle' }));
        line2.appendChild(svg('tspan', { class: 'lrpa-mid-name' }, `t${SUB[i]}′ = `));
        const vec = line2.appendChild(svg('tspan', { class: 'lrpa-mid-vec' }, ''));
        const line3 = pillG.appendChild(svg('text', { x: mx, y: my + 20, 'text-anchor': 'middle', class: 'lrpa-mid-r' }));
        const rPrefix = line3.appendChild(svg('tspan', {}, ''));
        const rVal = line3.appendChild(svg('tspan', { class: 'lrpa-rval' }, ''));
        return { g: pillG, eq, vec, rPrefix, rVal };
      });
      const centerTot = addText(L.text, g.centerTot[0], g.centerTot[1], '', 'lrpa-total');

      const softTrackG = L.track.appendChild(svg('g', { class: 'lrpa-soft-group' }));
      const softStreamG = L.stream.appendChild(svg('g', { class: 'lrpa-soft-group' }));
      const softFlowG = L.flow.appendChild(svg('g', { class: 'lrpa-soft-group' }));
      const softNodeG = L.nodes.appendChild(svg('g', { class: 'lrpa-soft-group' }));

      const bwdInPathsTall = [
        'M32,582 L30,677 Q30,683 36,683 L138,683',
        'M312,582 L312,584 Q312,587 308,587 L29,587 Q23,587 23,593 L23,702 Q23,708 29,708 L138,708',
        'M516,582 L516,589 Q516,592 512,592 L22,592 Q16,592 16,598 L16,727 Q16,733 22,733 L138,733'
      ];
      const bwdRows = [0, 1, 2].map((i) => {
        const ry = g.bwdRowY0 + i * 25;
        const inD = tall
          ? bwdInPathsTall[i]
          : `M757,${YS[i] - 10} C766,${YS[i] - 10} 766,${ry} 778,${ry}`;
        softTrackG.appendChild(svg('path', { d: inD, class: 'lrpa-soft-track' }));
        const inStream = softStreamG.appendChild(svg('path', { d: inD, class: 'lrpa-soft-stream' }));
        const inFlow = softFlowG.appendChild(svg('path', { d: inD, class: 'lrpa-bwd-flow' }));

        const rowG = L.nodes.appendChild(svg('g', { class: 'lrpa-srow' }));
        bindRow(rowG, i);
        rowG.appendChild(svg('rect', { x: g.bwdRowX, y: ry - 12, width: 188, height: 24, class: 'lrpa-brow-bg' }));
        rowG.appendChild(svg('text', { x: g.bwdRowTxtX, y: ry + 4, class: 'lrpa-box-sub', 'text-anchor': 'start' }, `t${SUB[i]}′`));

        const cells = g.bCols.map((cx) => ({
          r: rowG.appendChild(svg('rect', { x: cx - 19, y: ry - 10, width: 38, height: 20, rx: 5, class: 'lrpa-bcell-chip' })),
          t: rowG.appendChild(svg('text', { x: cx, y: ry + 3.5, 'text-anchor': 'middle', class: 'lrpa-bcell-val' }, ''))
        }));
        const rSum = rowG.appendChild(svg('text', { x: g.bwdRowSumX, y: ry + 4, 'text-anchor': 'middle', class: 'lrpa-box-sub' }, '0.00'));
        return { inStream, inFlow, rowG, cells, rSum };
      });

      const bwdOutPathsTall = [
        'M464,682 L580,682 Q586,682 586,688 L586,983 Q586,989 580,989 L174,989 Q168,989 168,995 L168,1002',
        'M464,712 L566,712 Q572,712 572,718 L572,989 Q572,995 566,995 L378,995 Q372,995 372,998 L372,1002',
        'M464,742 L552,742 Q558,742 558,748 L558,1002'
      ];
      const bwdSoftCards = g.softBusY.map((by, j) => {
        softTrackG.appendChild(svg('path', {
          d: `M${g.bwdBusX},${g.bwdBusMidY} C${g.bwdBusX + 8},${g.bwdBusMidY} ${g.bwdCardX - 9},${by} ${g.bwdCardX},${by}`,
          class: 'lrpa-soft-track'
        }));
        softNodeG.appendChild(svg('rect', { x: g.bwdCardX, y: by - 12, width: 120, height: 24, rx: 7, class: 'lrpa-weight-card' }));
        addText(L.text, g.bwdCardX + 10, by + 4, `to t${SUB[j]}`, 'lrpa-port-name is-sm', 'start');
        const amt = addText(L.text, g.bwdCardX + 112, by + 4, '', 'lrpa-weight-val', 'end');

        const laneX = 1142 - j * 10;
        const sd = tall
          ? bwdOutPathsTall[j]
          : pathD([[1104, by], [laneX, by], [laneX, YS[j] - 20], [1154, YS[j] - 20]]);
        softTrackG.appendChild(svg('path', { d: sd, class: 'lrpa-soft-track' }));
        const stream = softStreamG.appendChild(svg('path', { d: sd, class: 'lrpa-soft-stream' }));
        const flow = softFlowG.appendChild(svg('path', { d: sd, class: 'lrpa-bwd-flow' }));
        return { amt, stream, flow };
      });

      const cancelBanner = L.text.appendChild(svg('g', { class: 'lrpa-cancel-banner' }));
      const cancelRect = cancelBanner.appendChild(svg('rect', { x: g.cancelX, y: g.cancelY, width: 318, height: 22, rx: 11 }));
      const cancelTxt = cancelBanner.appendChild(svg('text', { x: g.cancelX + 159, y: g.cancelY + 15, 'text-anchor': 'middle' }, ''));

      const bwdCurves = [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
        (tall
          ? bezCurveVert([g.xs[i] + (j - 1) * 10, 836], [g.xs[j] + (i - 1) * 10, 1002], 66)
          : bezCurve([757, YS[i] + (j - 1) * 9], [1154, YS[j] + 16 + (i - 1) * 8], 145))));
      const bwdWires = bwdCurves.map((row) => row.map((p) => wire(L, p, false)));

      const rightCards = [0, 1, 2].map((j) => {
        const rx0 = tall ? g.xs[j] - g.rightW / 2 : 1154;
        const ry0 = tall ? 1056 : YS[j];
        const pad = tall ? 9 : 12;
        L.nodes.appendChild(svg('rect', { x: rx0, y: ry0 - 54, width: g.rightW, height: 108, rx: 12, class: 'lrpa-right-card' }));
        addText(L.text, rx0 + pad, ry0 - 33, `t${SUB[j]}`, 'lrpa-port-name', 'start');
        const tot = addText(L.text, rx0 + g.rightW - pad, ry0 - 33, '', 'lrpa-port-r', 'end');
        L.nodes.appendChild(svg('circle', { cx: rx0 + pad + 6, cy: ry0 - 18, r: 4, class: 'lrpa-dot-tok' }));
        const tok = addText(L.text, rx0 + pad + 15, ry0 - 14, '', 'lrpa-box-sub', 'start');
        L.nodes.appendChild(svg('circle', { cx: rx0 + pad + 6, cy: ry0, r: 4, class: 'lrpa-dot-soft' }));
        const soft = addText(L.text, rx0 + pad + 15, ry0 + 4, '', 'lrpa-box-sub', 'start');
        const vecR = addText(L.text, rx0 + pad, ry0 + 22, '', 'lrpa-box-sub', 'start');
        const rEq = addText(L.text, rx0 + pad, ry0 + 40, '', 'lrpa-box-sub', 'start');
        const rPrefix = rEq.appendChild(svg('tspan', {}, ''));
        const rVal = rEq.appendChild(svg('tspan', { class: 'lrpa-rval' }, ''));
        return { tot, tok, soft, vecR, rPrefix, rVal };
      });
      const rightTot = addText(L.text, g.rightTot[0], g.rightTot[1], '', 'lrpa-total');

      const updateDrop = (el, j, hi) => {
        if (hi === null) {
          el.pTxt.textContent = `× aᵢ${SUB[j]}`;
          el.line.setAttribute('d', tall ? `M${el.cx},258 L${el.cx},302` : `M${el.cx},168 L${el.cx},214`);
          el.mulG.style.display = 'none';
          return;
        }
        el.pTxt.textContent = `× ${fmt(hi.row.a[j], 2)}`;
        const pt = tall
          ? bezAtY(fwdCurves[hi.idx][j], 418 + (j - 1) * 14)
          : bezAtX(fwdCurves[hi.idx][j], el.cx);
        const px = pt[0].toFixed(1), py = pt[1].toFixed(1);
        el.line.setAttribute('d', tall
          ? `M${el.cx},258 L${el.cx},346 C${el.cx},376 ${px},360 ${px},${py}`
          : `M${el.cx},168 L${el.cx},${py}`);
        el.mulG.style.display = '';
        el.mulG.setAttribute('transform', `translate(${px}, ${py})`);
      };

      ui = {
        chips, softRows, dropEls, updateDrop, fwdWires, midTokens, centerTot,
        bwdSoftBox: tall ? bwdSoftBoxTall : bwdSoftBoxWide,
        softTrackG, softStreamG, softFlowG, softNodeG,
        bwdRows, bwdSoftCards, cancelRect, cancelTxt, bwdWires, rightCards, rightTot
      };
    };

    ruleBtns.forEach((b) => b.addEventListener('click', () => { st.mode = b.dataset.rule; update(); }));
    tauInp.addEventListener('input', () => { st.tau = Number(tauInp.value); update(); });

    new IntersectionObserver(([en]) =>
      root.classList.toggle('lrp-attn-paused', !en.isIntersecting)).observe(root);

    function update() {
      if (!ui) return;
      ui.chips.forEach((sync) => sync());
      ruleBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.rule === st.mode)));
      tauOut.textContent = fmt(st.tau);

      const dotMat = st.v.map((ti) => st.v.map((tj) => (ti[0] * tj[0] + ti[1] * tj[1]) * INV_SQRT2));
      const v1 = st.v.map((p) => p[0]);
      const v2 = st.v.map((p) => p[1]);

      const rows = dotMat.map((sRow) => {
        const s = sRow.map((x) => x / st.tau);
        const f1 = attnRow(v1, s, 1, 'cp');
        const f2 = attnRow(v2, s, 1, 'cp');
        const c1 = attnRow(v1, s, f1.y, st.mode);
        const c2 = attnRow(v2, s, f2.y, st.mode);
        return {
          sRow,
          a: c1.a,
          yVec: [f1.y, f2.y],
          R: f1.y + f2.y,
          rv1: c1.rv,
          rv2: c2.rv,
          rv: [0, 1, 2].map((j) => c1.rv[j] + c2.rv[j]),
          rs1: c1.rs,
          rs2: c2.rs,
          rs: [0, 1, 2].map((j) => c1.rs[j] + c2.rs[j])
        };
      });

      const hi = st.hover;
      const totR = sum(rows.map((r) => r.R));

      ui.softRows.forEach((sr, i) => {
        const r = rows[i];
        const dim = hi !== null && hi !== i;
        sr.rowG.classList.toggle('is-hover', hi === i);
        sr.rowG.classList.toggle('is-dim', dim);
        sr.sCells.forEach((sc, j) => { sc.t.textContent = fmt(r.sRow[j], 1); });
        sr.cells.forEach((c, j) => {
          const w = r.a[j];
          c.r.style.fill = `color-mix(in srgb, var(--fig-accent) ${Math.round(w * 85)}%, var(--fig-surface))`;
          c.t.textContent = fmt(w, 2);
          c.t.classList.toggle('is-light', w > 0.52);
        });
      });

      rows.forEach((r, i) => {
        const active = hi === null || hi === i;
        r.a.forEach((aij, j) => ui.fwdWires[i][j](aij, aij * Math.hypot(st.v[j][0], st.v[j][1]), active));
      });
      const hiInfo = hi === null ? null : { idx: hi, row: rows[hi] };
      ui.dropEls.forEach((el, j) => ui.updateDrop(el, j, hiInfo));

      rows.forEach((r, i) => {
        ui.midTokens[i].eq.textContent = `${fmt(r.a[0])} t₁ + ${fmt(r.a[1])} t₂ + ${fmt(r.a[2])} t₃`;
        ui.midTokens[i].vec.textContent = fmtVec(r.yVec);
        const [lhs, rhs] = fmtSumParts(r.yVec[0], r.yVec[1], r.R);
        ui.midTokens[i].rPrefix.textContent = `R = ${lhs}`;
        ui.midTokens[i].rVal.textContent = rhs;
        ui.midTokens[i].rVal.classList.toggle('is-pos', r.R > 0.005);
        ui.midTokens[i].rVal.classList.toggle('is-neg', r.R < -0.005);
        ui.midTokens[i].g.classList.toggle('is-hover', hi === i);
        ui.midTokens[i].g.classList.toggle('is-dim', hi !== null && hi !== i);
      });
      ui.centerTot.textContent = `total R = ${fmt(totR)}`;

      const activeRows = hi === null ? rows : [rows[hi]];
      const shownSoft = [0, 1, 2].map((j) => sum(activeRows.map((r) => r.rs[j])));
      const allTok1 = [0, 1, 2].map((j) => sum(rows.map((r) => r.rv1[j])));
      const allTok2 = [0, 1, 2].map((j) => sum(rows.map((r) => r.rv2[j])));
      const allSoft1 = [0, 1, 2].map((j) => sum(rows.map((r) => r.rs1[j])));
      const allSoft2 = [0, 1, 2].map((j) => sum(rows.map((r) => r.rs2[j])));
      const allTok = [0, 1, 2].map((j) => allTok1[j] + allTok2[j]);
      const allSoft = [0, 1, 2].map((j) => allSoft1[j] + allSoft2[j]);
      const allNet = [0, 1, 2].map((j) => allTok[j] + allSoft[j]);
      const sumNet = sum(allNet);

      const isFrozen = st.mode === 'cp';
      const isGrad = st.mode === 'grad';
      ui.bwdSoftBox.classList.toggle('is-frozen', isFrozen);
      ui.bwdSoftBox.classList.toggle('is-grad', isGrad);
      ui.bwdSoftBox.classList.toggle('is-attnlrp', st.mode === 'ah');
      ui.softTrackG.classList.toggle('is-frozen', isFrozen);
      ui.softStreamG.classList.toggle('is-frozen', isFrozen);
      ui.softFlowG.classList.toggle('is-frozen', isFrozen);
      ui.softNodeG.classList.toggle('is-frozen', isFrozen);

      ui.bwdRows.forEach((br, i) => {
        const r = rows[i];
        const dim = hi !== null && hi !== i;
        br.rowG.classList.toggle('is-hover', hi === i);
        br.rowG.classList.toggle('is-dim', dim);

        const showIn = !isFrozen && (hi === null || hi === i);
        const wIn = showIn ? Math.min(Math.abs(r.R) * 12, 14) : 0;
        br.inStream.style.strokeWidth = `${wIn}px`;
        br.inStream.classList.toggle('is-neg', r.R < -1e-6);
        br.inFlow.style.strokeWidth = `${Math.min(2.2, wIn * 0.34)}px`;
        br.inFlow.classList.toggle('is-dim', !showIn || wIn <= 1.8);

        let rSum = 0;
        br.cells.forEach((c, j) => {
          const val = isFrozen ? 0 : r.rs[j];
          rSum += val;
          c.t.textContent = isFrozen ? '0.00' : fmtS(val);
          c.t.classList.toggle('is-pos', !isFrozen && val > 0.005);
          c.t.classList.toggle('is-neg', !isFrozen && val < -0.005);
          if (isFrozen) {
            c.r.style.fill = 'var(--fig-surface)';
          } else {
            const pct = Math.min(Math.round(Math.abs(val) * 28), 55);
            const tint = val >= 0 ? 'var(--lrpw-rel)' : 'var(--lrpw-neg)';
            c.r.style.fill = `color-mix(in srgb, ${tint} ${pct}%, var(--fig-surface))`;
          }
        });
        br.rSum.textContent = isFrozen ? '0.00' : fmt(rSum);
      });

      ui.bwdSoftCards.forEach((sc, j) => {
        const q = shownSoft[j];
        sc.amt.textContent = fmtS(q);
        sc.amt.classList.toggle('is-pos', q > 0.005);
        sc.amt.classList.toggle('is-neg', q < -0.005);
        const w = isFrozen ? 0 : Math.min(Math.abs(q) * 14, 18);
        sc.stream.style.strokeWidth = `${w}px`;
        sc.stream.classList.toggle('is-neg', q < -1e-6);
        sc.flow.style.strokeWidth = `${Math.min(2.2, w * 0.34)}px`;
        sc.flow.classList.toggle('is-dim', isFrozen || w <= 1.8);
      });

      const posSoft = sum(shownSoft.map((v) => Math.max(0, v)));
      const negSoft = sum(shownSoft.map((v) => Math.max(0, -v)));
      const netSoft = sum(shownSoft);
      if (isFrozen) {
        ui.cancelRect.style.stroke = 'var(--fig-accent)';
        ui.cancelTxt.innerHTML = 'Frozen: weights receive no gradient → <tspan class="is-pos">0.00</tspan>';
      } else if (isGrad) {
        ui.cancelRect.style.stroke = 'var(--lrp-warn)';
        const prefix = hi === null ? 'Every row cancels' : `Row t${SUB[hi]}′ cancels`;
        ui.cancelTxt.innerHTML = `${prefix}: <tspan class="is-pos">+${fmt(posSoft)}</tspan> <tspan class="is-neg">−${fmt(negSoft)}</tspan> = ${fmt(netSoft)}`;
      } else {
        ui.cancelRect.style.stroke = 'var(--fig-accent)';
        ui.cancelTxt.innerHTML = `AttnLRP z⁺ passes <tspan class="is-pos">${fmtS(netSoft)}</tspan> via keys/queries`;
      }

      rows.forEach((r, i) => {
        const active = hi === null || hi === i;
        r.rv.forEach((rv, j) => ui.bwdWires[i][j](rv, active));
      });

      ui.rightCards.forEach((rc, j) => {
        rc.tot.textContent = fmtS(allNet[j]);
        rc.tot.classList.toggle('is-pos', allNet[j] > 0.005);
        rc.tot.classList.toggle('is-neg', allNet[j] < -0.005);
        rc.tok.textContent = `tok: ${fmtVec([allTok1[j], allTok2[j]])} → ${fmtS(allTok[j])}`;
        rc.soft.textContent = `soft: ${fmtVec([allSoft1[j], allSoft2[j]])} → ${fmtS(allSoft[j])}`;
        rc.vecR.textContent = `vec R = ${fmtVec([allTok1[j] + allSoft1[j], allTok2[j] + allSoft2[j]])}`;
        const [lhs, rhs] = fmtSumParts(allTok1[j] + allSoft1[j], allTok2[j] + allSoft2[j], allNet[j]);
        rc.rPrefix.textContent = `R = ${lhs}`;
        rc.rVal.textContent = rhs;
        rc.rVal.classList.toggle('is-pos', allNet[j] > 0.005);
        rc.rVal.classList.toggle('is-neg', allNet[j] < -0.005);
      });
      ui.rightTot.textContent = `total R = ${fmt(sumNet)}`;
      ui.rightTot.classList.toggle('lrp-nt-warn', Math.abs(sumNet - totR) > 0.01);

      const t3Sum = st.v[2][0] + st.v[2][1];
      if (st.mode === 'cp') {
        note.innerHTML = `<b>Freeze weights (Conservative propagation):</b> Each token is a 2D vector <code>tⱼ ∈ ℝ²</code>, so scaled dot products <code>(tᵢ·t₃)/√2</code> give highest weight to <code>t₃ = ${fmtVec(st.v[2], 1)}</code> (coordinate sum <code>${fmtS(t3Sum)}</code>). Freezing those weights sends <b>0.00</b> into Softmax and attributes each coordinate by <code>aᵢⱼ tⱼ,c</code>—summing both coordinates gives <code>t₃</code> <b>${fmtS(allNet[2])}</b>.`;
      } else if (st.mode === 'grad') {
        note.innerHTML = `<b>Gradient fallback:</b> Each updated vector <code>tᵢ′</code> mixes both coordinates with weight <code>aᵢⱼ</code>. Along the <b>Token path</b>, relevance flows by <code>aᵢⱼ Rᵢ</code> regardless of <code>t₃</code>'s coordinates—sending <b>${fmtS(allTok[2])}</b> to <code>t₃ = ${fmtVec(st.v[2], 1)}</code> while the <b>Softmax path</b> pushes <b>${fmtS(allSoft[2])}</b> (net <b>${fmtS(allNet[2])}</b>, conserving total <b>${fmt(sumNet)}</b>).`;
      } else {
        note.innerHTML = `<b>AttnLRP (50/50 split + Softmax z⁺):</b> Splits each coordinate's outgoing relevance equally between the Token path and the Softmax <code>z⁺</code> rule. <code>t₃ = ${fmtVec(st.v[2], 1)}</code> receives <b>${fmtS(allTok[2])}</b> on the token path and <b>${fmtS(allSoft[2])}</b> on the softmax path.`;
      }
    }

    const fit = () => {
      const wantTall = (root.clientWidth || window.innerWidth) < 820;
      if (wantTall === isTall) return;
      isTall = wantTall;
      buildScene(isTall);
      update();
    };
    fit();
    new ResizeObserver(fit).observe(root);
  };

  const init = () => document.querySelectorAll('[data-lrp-attn]').forEach((root) => {
    if (root.dataset.ready) return;
    root.dataset.ready = 'true';
    initFigure(root);
  });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

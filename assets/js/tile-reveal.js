(() => {
  "use strict";

  if (window.TileReveal) return;

  const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)");

  const px = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const ms = (value, fallback) => {
    if (!value) return fallback;
    const s = String(value).trim();
    const n = parseFloat(s);
    if (!Number.isFinite(n)) return fallback;
    return s.endsWith("ms") ? n : s.endsWith("s") ? n * 1000 : n;
  };

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const smooth = (t) => t * t * (3 - 2 * t);

  const makeBezierLut = (x1, y1, x2, y2, size = 256) => {
    const lut = new Float32Array(size);
    const sampleX = (u) => ((3 * x1 - 3 * x2 + 1) * u * u * u) + ((3 * x2 - 6 * x1) * u * u) + (3 * x1 * u);
    const sampleY = (u) => ((3 * y1 - 3 * y2 + 1) * u * u * u) + ((3 * y2 - 6 * y1) * u * u) + (3 * y1 * u);
    const sampleDX = (u) => (3 * (3 * x1 - 3 * x2 + 1) * u * u) + (2 * (3 * x2 - 6 * x1) * u) + (3 * x1);

    for (let i = 0; i < size; i++) {
      const x = i / (size - 1);
      if (x <= 0) {
        lut[i] = 0;
        continue;
      }
      if (x >= 1) {
        lut[i] = 1;
        continue;
      }
      let u = x;
      for (let k = 0; k < 8; k++) {
        const err = sampleX(u) - x;
        if (Math.abs(err) < 1e-6) break;
        const d = sampleDX(u);
        if (Math.abs(d) < 1e-6) break;
        u = clamp01(u - err / d);
      }
      lut[i] = sampleY(u);
    }
    return (t) => {
      if (t <= 0) return 0;
      if (t >= 1) return 1;
      const pos = t * (size - 1);
      const idx = pos | 0;
      if (idx >= size - 1) return 1;
      return lut[idx] + (lut[idx + 1] - lut[idx]) * (pos - idx);
    };
  };

  // Matches CSS `cubic-bezier(.3, .9, .3, 1)` for transform and `ease` for opacity
  const easeMove = makeBezierLut(0.3, 0.9, 0.3, 1.0);
  const easeOpacity = makeBezierLut(0.25, 0.1, 0.25, 1.0);

  const TILE_CAP = 14;
  const MAX_TILES = 64;
  const MAX_CANVAS = 640;
  const HOLE = 0.45;
  const MOVE = 0.3;
  const PRESSED = 2.4;
  const HOLD = 180;
  const SLOP = 10;
  const TAP = 140;

  const activeMaps = new Set();
  let rafId = 0;

  const stepAll = (now) => {
    rafId = 0;
    activeMaps.forEach((map) => {
      if (!map.owner.stepMap(map, now)) {
        activeMaps.delete(map);
      }
    });
    if (activeMaps.size > 0) {
      rafId = requestAnimationFrame(stepAll);
    }
  };

  const wake = (map) => {
    map.animating = true;
    activeMaps.add(map);
    if (!rafId) rafId = requestAnimationFrame(stepAll);
  };

  const sleep = (map) => {
    map.animating = false;
    activeMaps.delete(map);
  };

  class TileReveal {
    constructor(root, { prefix = "fg", autoArm = true, getCols = null } = {}) {
      this.root = root;
      this.prefix = prefix;
      this.cls = {
        stack: `${prefix}-stack`,
        overlay: `${prefix}-overlay`,
        tiles: `${prefix}-tiles`,
        tiled: `${prefix}-tiled`,
        armed: `${prefix}-armed`,
        grid: `${prefix}-grid`,
      };
      this.getCols = getCols;
      this.maps = [];
      this.byStack = new Map();

      this.geometry();
      this.bindMaps(autoArm);
      // A resized canvas is rebuilt on the next interaction at the new size.
      addEventListener("resize", () => {
        this.geometry();
        this.maps.forEach((map) => {
          map.width = 0;
          this.strip(map);
        });
      }, { passive: true });
    }

    geometry() {
      const p = this.prefix;
      const style = getComputedStyle(this.root);
      this.geo = {
        reach: px(style.getPropertyValue(`--${p}-reach`), 0.46),
        depth: px(style.getPropertyValue(`--${p}-depth`), 230),
        spread: px(style.getPropertyValue(`--${p}-spread`), 45),
        perspective: px(style.getPropertyValue(`--${p}-perspective`), 620),
        bleed: px(style.getPropertyValue(`--${p}-tile-bleed`), 1),
        tileMove: ms(style.getPropertyValue(`--${p}-tile-move`), 140),
        wave: ms(style.getPropertyValue(`--${p}-wave`), 420),
        settleHold: ms(style.getPropertyValue(`--${p}-settle-hold`), 50),
        settleStep: ms(style.getPropertyValue(`--${p}-settle-step`), p === "fg" ? 130 : 90),
        settleWave: ms(style.getPropertyValue(`--${p}-settle-wave`), 420),
        settleMove: ms(style.getPropertyValue(`--${p}-settle-move`), 200),
      };
    }

    bindMaps(autoArm) {
      this.pressable = () => !calm.matches;
      this.wanted = () => fine.matches && this.pressable();

      [...this.root.querySelectorAll(`.${this.cls.stack}`)].forEach((stack) => {
        const overlay = stack.querySelector(`.${this.cls.overlay}`);
        if (!overlay) return;
        const map = {
          owner: this,
          stack,
          overlay,
          col: 0,
          canvas: null,
          ctx: null,
          n: 0,
          crisp: true,
          width: 0,
          curTx: null,
          curTy: null,
          curScale: null,
          curOp: null,
          fromTx: null,
          fromTy: null,
          fromScale: null,
          fromOp: null,
          toTx: null,
          toTy: null,
          toScale: null,
          toOp: null,
          startAt: null,
          dur: null,
          live: new Set(),
          at: null,
          held: false,
          busy: false,
          arrived: false,
          animating: false,
          onFinish: null,
          onLoad: null,
          origin: null,
          timer: 0,
          settle: 0,
          idle: 0,
          frame: 0,
        };
        this.maps.push(map);
        this.byStack.set(stack, map);
      });

      if (this.root.dataset.reveal === "continuous") this.sheet();
      else this.maps.forEach((map) => this.listen(map));

      this.settleIn(autoArm);
    }

    build(map) {
      clearTimeout(map.idle);
      const overlay = map.overlay;
      if (!overlay.complete || !overlay.naturalWidth) return false;

      // Called on every pointer move, so an existing canvas returns immediately.
      if (map.canvas) return true;

      const crisp = overlay.dataset.rendering !== "auto";
      const width = overlay.naturalWidth || 0;
      // Same tile count as the old DOM grid: one tile per pixel for crisp maps, capped otherwise.
      const wanted = parseInt(map.stack.dataset.tiles, 10)
        || (crisp ? width : Math.min(width, TILE_CAP));
      const n = Math.min(Math.max(wanted, 0), MAX_TILES);
      if (n < 2) return false;

      const cssW = map.width || map.stack.clientWidth || 240;
      map.width = cssW;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = Math.min(MAX_CANVAS, Math.max(224, Math.round(cssW * dpr)));

      const canvas = document.createElement("canvas");
      canvas.className = this.cls.tiles;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");

      const count = n * n;
      map.n = n;
      map.crisp = crisp;
      map.canvas = canvas;
      map.ctx = ctx;
      map.curTx = new Float32Array(count);
      map.curTy = new Float32Array(count);
      map.curScale = new Float32Array(count);
      map.curOp = new Float32Array(count);
      map.fromTx = new Float32Array(count);
      map.fromTy = new Float32Array(count);
      map.fromScale = new Float32Array(count);
      map.fromOp = new Float32Array(count);
      map.toTx = new Float32Array(count);
      map.toTy = new Float32Array(count);
      map.toScale = new Float32Array(count);
      map.toOp = new Float32Array(count);
      map.startAt = new Float64Array(count);
      map.dur = new Float32Array(count);

      this.resetTilesInstant(map, 0, 0, this.geo.bleed, 1);
      this.renderMap(map);
      map.stack.append(canvas);
      this.swap(map, true);
      return true;
    }

    resetTilesInstant(map, tx, ty, scale, op) {
      const count = map.n * map.n;
      const now = performance.now();
      map.curTx.fill(tx);
      map.curTy.fill(ty);
      map.curScale.fill(scale);
      map.curOp.fill(op);
      map.fromTx.fill(tx);
      map.fromTy.fill(ty);
      map.fromScale.fill(scale);
      map.fromOp.fill(op);
      map.toTx.fill(tx);
      map.toTy.fill(ty);
      map.toScale.fill(scale);
      map.toOp.fill(op);
      map.startAt.fill(now);
      map.dur.fill(0);
      map.live.clear();
    }

    setTile(map, i, tx, ty, scale, op, now, duration, delay = 0) {
      if (
        delay === 0 &&
        Math.abs(map.toTx[i] - tx) < 1e-4 &&
        Math.abs(map.toTy[i] - ty) < 1e-4 &&
        Math.abs(map.toScale[i] - scale) < 1e-4 &&
        Math.abs(map.toOp[i] - op) < 1e-4
      ) {
        return;
      }
      map.fromTx[i] = map.curTx[i];
      map.fromTy[i] = map.curTy[i];
      map.fromScale[i] = map.curScale[i];
      map.fromOp[i] = map.curOp[i];
      map.toTx[i] = tx;
      map.toTy[i] = ty;
      map.toScale[i] = scale;
      map.toOp[i] = op;
      map.startAt[i] = now + delay;
      map.dur[i] = duration;
    }

    stepMap(map, now) {
      if (!map.canvas) return false;
      const count = map.n * map.n;
      const {
        curTx, curTy, curScale, curOp,
        fromTx, fromTy, fromScale, fromOp,
        toTx, toTy, toScale, toOp,
        startAt, dur,
      } = map;

      let moving = false;
      for (let i = 0; i < count; i++) {
        const d = dur[i];
        if (
          curTx[i] === toTx[i] &&
          curTy[i] === toTy[i] &&
          curScale[i] === toScale[i] &&
          curOp[i] === toOp[i] &&
          now >= startAt[i] + d
        ) {
          continue;
        }
        const elapsed = now - startAt[i];
        if (elapsed <= 0) {
          moving = true;
          continue;
        }
        if (elapsed >= d || d <= 0) {
          curTx[i] = toTx[i];
          curTy[i] = toTy[i];
          curScale[i] = toScale[i];
          curOp[i] = toOp[i];
        } else {
          const t = elapsed / d;
          const em = easeMove(t);
          const eo = easeOpacity(t);
          curTx[i] = fromTx[i] + (toTx[i] - fromTx[i]) * em;
          curTy[i] = fromTy[i] + (toTy[i] - fromTy[i]) * em;
          curScale[i] = fromScale[i] + (toScale[i] - fromScale[i]) * em;
          curOp[i] = fromOp[i] + (toOp[i] - fromOp[i]) * eo;
          moving = true;
        }
      }

      this.renderMap(map);

      if (!moving) {
        map.animating = false;
        if (map.onFinish) {
          const fn = map.onFinish;
          map.onFinish = null;
          fn();
        }
        if (!map.at && !map.held && !map.busy) {
          this.rested(map, 600);
        }
        return false;
      }
      return true;
    }

    renderMap(map) {
      const { canvas, ctx, overlay, n } = map;
      if (!canvas || !ctx || !overlay.complete || !overlay.naturalWidth) return;

      const W = canvas.width;
      const H = canvas.height;
      const imgW = overlay.naturalWidth;
      const imgH = overlay.naturalHeight;

      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = !map.crisp;

      const cellW = W / n;
      const cellH = H / n;
      const sw = imgW / n;
      const sh = imgH / n;
      const { curTx, curTy, curScale, curOp } = map;

      for (let r = 0; r < n; r++) {
        const sy = r * sh;
        const y0 = Math.round(r * cellH);
        const y1 = Math.round((r + 1) * cellH);
        const h0 = y1 - y0;
        const cyBase = (y0 + y1) * 0.5;

        for (let c = 0; c < n; c++) {
          const i = r * n + c;
          const op = curOp[i];
          if (op <= 0.001) continue;

          const sx = c * sw;
          const x0 = Math.round(c * cellW);
          const x1 = Math.round((c + 1) * cellW);
          const w0 = x1 - x0;

          const tx = curTx[i];
          const ty = curTy[i];
          const s = curScale[i];

          ctx.globalAlpha = op >= 0.999 ? 1 : op;

          if (tx === 0 && ty === 0 && Math.abs(s - 1) < 0.001) {
            ctx.drawImage(overlay, sx, sy, sw, sh, x0, y0, w0, h0);
          } else {
            const dw = w0 * s;
            const dh = h0 * s;
            const dx = (x0 + x1) * 0.5 + tx * cellW - dw * 0.5;
            const dy = cyBase + ty * cellH - dh * 0.5;
            ctx.drawImage(overlay, sx, sy, sw, sh, dx, dy, dw, dh);
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    // Swaps between the canvas and the static <img> without letting the overlay's
    // opacity transition flash. The double rAF restores the transition one frame
    // after the swap has been styled, so no forced reflow is needed.
    swap(map, tiled) {
      const overlay = map.overlay;
      overlay.style.transition = "none";
      map.stack.classList.toggle(this.cls.tiled, tiled);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        overlay.style.transition = "";
      }));
    }

    strip(map) {
      if (!map.canvas || map.held || map.busy) return;
      clearTimeout(map.idle);
      sleep(map);
      map.canvas.remove();
      map.canvas = null;
      map.ctx = null;
      map.n = 0;
      map.live.clear();
      this.swap(map, false);
    }

    detachCanvas(map) {
      clearTimeout(map.idle);
      sleep(map);
      const canvas = map.canvas;
      if (canvas) {
        map.canvas = null;
        map.ctx = null;
        map.n = 0;
        map.live.clear();
        this.swap(map, false);
      }
      return canvas;
    }

    cancel(map) {
      clearTimeout(map.timer);
      clearTimeout(map.settle);
      clearTimeout(map.idle);
      if (map.frame) {
        cancelAnimationFrame(map.frame);
        map.frame = 0;
      }
      if (map.onLoad) {
        map.overlay.removeEventListener("load", map.onLoad);
        map.onLoad = null;
      }
      map.onFinish = null;
      map.held = false;
      map.busy = false;
      sleep(map);
    }

    rested(map, delay = 800) {
      clearTimeout(map.idle);
      map.idle = setTimeout(() => {
        if (map.at || map.held || map.busy || map.animating) return;
        this.strip(map);
      }, delay);
    }

    rest(map) {
      if (!map.canvas) return;
      const now = performance.now();
      const dur = this.geo.tileMove;
      const bleed = this.geo.bleed;
      const count = map.n * map.n;
      for (let i = 0; i < count; i++) {
        this.setTile(map, i, 0, 0, bleed, 1, now, dur, 0);
      }
      map.live.clear();
      wake(map);
    }

    shrink(z) {
      const p = this.geo.perspective;
      return (this.geo.bleed * p) / (p + z);
    }

    receded(u, ux, uy) {
      const opacity = smooth(clamp01((u - HOLE) / (1 - HOLE)));
      const k = u < HOLE ? 1 - u : (1 - HOLE) * smooth(clamp01((MOVE - opacity) / MOVE));
      const out = (k * this.geo.spread) / 100;
      return [
        ux * out,
        uy * out,
        this.shrink(this.geo.depth * k * k),
        opacity,
      ];
    }

    cleared(ux, uy) {
      const out = (this.geo.spread * 3.2) / 100;
      return [
        ux * out,
        uy * out,
        this.shrink(this.geo.depth * PRESSED),
        0,
      ];
    }

    hover(map, x, y, w, h) {
      if (map.busy || !map.canvas) return;
      const { n } = map;
      const reach = this.geo.reach * n;
      const cx = (x / w) * n - 0.5;
      const cy = (y / h) * n - 0.5;
      const next = new Set();
      const c0 = Math.max(0, Math.floor(cx - reach));
      const c1 = Math.min(n - 1, Math.ceil(cx + reach));
      const r0 = Math.max(0, Math.floor(cy - reach));
      const r1 = Math.min(n - 1, Math.ceil(cy + reach));
      const now = performance.now();
      const dur = this.geo.tileMove;

      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c - cx;
          const dy = r - cy;
          const d = Math.hypot(dx, dy);
          if (d >= reach) continue;
          const i = r * n + c;
          const [tx, ty, scale, op] = this.receded(d / reach, dx / (d || 1), dy / (d || 1));
          this.setTile(map, i, tx, ty, scale, op, now, dur, 0);
          next.add(i);
          map.live.delete(i);
        }
      }
      const bleed = this.geo.bleed;
      map.live.forEach((i) => this.setTile(map, i, 0, 0, bleed, 1, now, dur, 0));
      map.live = next;
      wake(map);
    }

    wave(map, out) {
      if (!map.canvas) return;
      const { n, origin } = map;
      const reach = this.geo.reach * n;
      const back = !out && map.at && map.at.inside ? map.at : null;
      const far = Math.max(
        Math.hypot(origin.cx + 0.5, origin.cy + 0.5),
        Math.hypot(n - 0.5 - origin.cx, origin.cy + 0.5),
        Math.hypot(origin.cx + 0.5, n - 0.5 - origin.cy),
        Math.hypot(n - 0.5 - origin.cx, n - 0.5 - origin.cy),
      ) || 1;

      const live = new Set();
      const now = performance.now();
      const dur = this.geo.tileMove;
      const waveMs = this.geo.wave;
      const bleed = this.geo.bleed;
      const count = n * n;

      for (let i = 0; i < count; i++) {
        const r = (i / n) | 0;
        const c = i % n;
        const dx = c - origin.cx;
        const dy = r - origin.cy;
        const d = Math.hypot(dx, dy);
        const norm = d / far;
        const delay = (out ? norm : 1 - norm) * waveMs;

        let tx = 0;
        let ty = 0;
        let scale = bleed;
        let op = 1;

        if (out) {
          [tx, ty, scale, op] = this.cleared(dx / (d || 1), dy / (d || 1));
        } else if (back) {
          const bx = c - back.cx;
          const by = r - back.cy;
          const bd = Math.hypot(bx, by);
          if (bd < reach) {
            [tx, ty, scale, op] = this.receded(bd / reach, bx / (bd || 1), by / (bd || 1));
            live.add(i);
          }
        }
        this.setTile(map, i, tx, ty, scale, op, now, dur, delay);
      }
      if (!out) map.live = live;
      wake(map);
    }

    press(map, x, y, w, h) {
      clearTimeout(map.timer);
      if (map.frame) {
        cancelAnimationFrame(map.frame);
        map.frame = 0;
      }
      map.onFinish = null;
      map.live.clear();
      map.held = true;
      map.busy = true;
      map.origin = { cx: (x / w) * map.n - 0.5, cy: (y / h) * map.n - 0.5 };
      this.wave(map, true);
    }

    release(map) {
      if (!map.held) return;
      map.held = false;
      if (!map.canvas) {
        map.busy = false;
        return;
      }
      map.onFinish = () => {
        clearTimeout(map.timer);
        map.busy = false;
      };
      this.wave(map, false);
      map.timer = setTimeout(() => {
        map.busy = false;
        if (!map.at && !map.held) this.rested(map);
      }, this.geo.tileMove + this.geo.wave + 80);
    }

    sheet() {
      const area = this.root.querySelector(`.${this.cls.grid}`) || this.root;
      let frame = 0;
      let at = null;

      const each = (fn) => this.maps.forEach((map) => {
        if (!this.build(map)) return;
        const b = map.stack.getBoundingClientRect();
        map.width = b.width;
        fn(map, { x: at.x - b.left, y: at.y - b.top, w: b.width, h: b.height });
      });

      const draw = () => {
        frame = 0;
        if (!at) return;
        each((map, p) => {
          map.at = {
            cx: (p.x / p.w) * map.n - 0.5,
            cy: (p.y / p.h) * map.n - 0.5,
            inside: p.x >= 0 && p.x <= p.w && p.y >= 0 && p.y <= p.h,
          };
          if (!map.busy) this.hover(map, p.x, p.y, p.w, p.h);
        });
      };

      area.addEventListener("pointermove", (event) => {
        if (event.pointerType === "touch" || !this.wanted()) return;
        at = { x: event.clientX, y: event.clientY };
        if (!frame) frame = requestAnimationFrame(draw);
      });

      area.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "touch") return;
        at = null;
        this.maps.forEach((map) => {
          map.at = null;
          this.release(map);
          if (!map.held) this.rest(map);
          this.rested(map);
        });
      });

      area.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "touch" || event.button !== 0 || !this.pressable()) return;
        event.preventDefault();
        try {
          area.setPointerCapture(event.pointerId);
        } catch (err) { /* ignore */ }
        at = { x: event.clientX, y: event.clientY };
        each((map, p) => this.press(map, p.x, p.y, p.w, p.h));
      });

      ["pointerup", "pointercancel"].forEach((name) => {
        area.addEventListener(name, (event) => {
          if (event.pointerType !== "touch") this.maps.forEach((map) => this.release(map));
        });
      });

      this.touch(area, (x, y) => {
        at = { x, y };
        each((map, p) => this.press(map, p.x, p.y, p.w, p.h));
        at = null;
      }, () => this.maps.forEach((map) => this.release(map)));
    }

    listen(map) {
      const { stack } = map;
      const local = (event) => {
        const b = stack.getBoundingClientRect();
        map.width = b.width;
        return {
          x: event.clientX - b.left,
          y: event.clientY - b.top,
          w: b.width,
          h: b.height,
        };
      };

      stack.addEventListener("pointermove", (event) => {
        if (event.pointerType === "touch" || !this.wanted() || !this.build(map)) return;
        const p = local(event);
        map.at = {
          cx: (p.x / p.w) * map.n - 0.5,
          cy: (p.y / p.h) * map.n - 0.5,
          inside: p.x >= 0 && p.x <= p.w && p.y >= 0 && p.y <= p.h,
        };
        if (map.busy || map.frame) return;
        map.frame = requestAnimationFrame(() => {
          map.frame = 0;
          this.hover(map, p.x, p.y, p.w, p.h);
        });
      });

      stack.addEventListener("pointerleave", (event) => {
        if (event.pointerType === "touch") return;
        map.at = null;
        this.release(map);
        if (!map.held) this.rest(map);
        this.rested(map);
      });

      stack.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "touch" || event.button !== 0) return;
        if (!this.pressable() || !this.build(map)) return;
        event.preventDefault();
        try {
          stack.setPointerCapture(event.pointerId);
        } catch (err) { /* ignore */ }
        const p = local(event);
        this.press(map, p.x, p.y, p.w, p.h);
      });

      ["pointerup", "pointercancel"].forEach((name) => {
        stack.addEventListener(name, (event) => {
          if (event.pointerType !== "touch") this.release(map);
        });
      });

      this.touch(stack, (x, y) => {
        const b = stack.getBoundingClientRect();
        map.width = b.width;
        if (!this.build(map)) return;
        this.press(map, x - b.left, y - b.top, b.width, b.height);
      }, () => this.release(map));
    }

    touch(area, press, release) {
      let down = null;
      let tap = 0;
      const start = () => {
        down.pressed = true;
        clearTimeout(tap);
        press(down.x, down.y);
      };
      const end = () => {
        clearTimeout(down.timer);
        down = null;
      };

      area.addEventListener("pointerdown", (event) => {
        if (event.pointerType !== "touch" || !event.isPrimary || !this.pressable()) return;
        if (down) end();
        down = { id: event.pointerId, x: event.clientX, y: event.clientY, pressed: false };
        down.timer = setTimeout(start, HOLD);
      });

      area.addEventListener("pointermove", (event) => {
        if (!down || down.pressed || event.pointerId !== down.id) return;
        if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > SLOP) end();
      });

      area.addEventListener("pointerup", (event) => {
        if (!down || event.pointerId !== down.id) return;
        if (down.pressed) release();
        else {
          start();
          tap = setTimeout(release, TAP);
        }
        end();
      });

      area.addEventListener("pointercancel", (event) => {
        if (!down || event.pointerId !== down.id) return;
        if (down.pressed) release();
        end();
      });

      area.addEventListener("contextmenu", (event) => {
        if (down) event.preventDefault();
      });
    }

    settleIn(autoArm) {
      if (calm.matches || !("IntersectionObserver" in window)) return;
      const cols = this.getCols ? this.getCols() : 1;
      this.maps.forEach((map, i) => {
        map.col = i % cols;
      });

      this.seen = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          this.seen.unobserve(entry.target);
          const map = this.byStack.get(entry.target);
          if (map) {
            if (entry.boundingClientRect && entry.boundingClientRect.width > 0) {
              map.width = entry.boundingClientRect.width;
            }
            this.arrive(map);
          }
        });
      }, { threshold: 0.4 });

      this.gone = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) return;
          const map = this.byStack.get(entry.target);
          if (map) this.strip(map);
        });
      }, { rootMargin: "300px" });

      if (autoArm) {
        this.armAll();
      }
    }

    arm(map) {
      if (!this.seen || !map) return;
      this.cancel(map);
      this.seen.unobserve(map.stack);
      map.stack.classList.add(this.cls.armed);
      map.live.clear();
      this.seen.observe(map.stack);
    }

    armAll() {
      this.maps.forEach((map) => this.arm(map));
    }

    arrive(map) {
      if (!map) return;
      if (!this.build(map)) {
        if (map.onLoad) map.overlay.removeEventListener("load", map.onLoad);
        map.onLoad = () => {
          map.onLoad = null;
          this.arrive(map);
        };
        map.overlay.addEventListener("load", map.onLoad, { once: true });
        map.settle = setTimeout(() => map.stack.classList.remove(this.cls.armed), 3000);
        return;
      }

      const { stack, n } = map;
      clearTimeout(map.settle);
      const far = Math.hypot(n - 1, n - 1) || 1;
      const fromScale = this.shrink(this.geo.depth);

      this.resetTilesInstant(map, 0, 0, fromScale, 0);
      this.renderMap(map);
      stack.classList.remove(this.cls.armed);

      const now = performance.now();
      const baseDelay = this.geo.settleHold + map.col * this.geo.settleStep;
      const waveMs = this.geo.settleWave;
      const moveMs = this.geo.settleMove;
      const bleed = this.geo.bleed;
      const count = n * n;

      for (let i = 0; i < count; i++) {
        const dNorm = Math.hypot(i % n, (i / n) | 0) / far;
        this.setTile(map, i, 0, 0, bleed, 1, now, moveMs, baseDelay + dNorm * waveMs);
      }

      map.arrived = true;
      wake(map);
      if (this.gone) this.gone.observe(stack);
    }
  }

  window.TileReveal = TileReveal;
})();

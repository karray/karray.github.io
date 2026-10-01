(() => {
  "use strict";

  if (window.HeatmapGrid) return;

  const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
  const px = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  const smooth = (t) => t * t * (3 - 2 * t);

  const TILE_CAP = 14;

  const HOLE = 0.45;

  const MOVE = 0.3;

  const PRESSED = 2.4;

  const SETTLE = 900;

  const HOLD = 180;
  const SLOP = 10;
  const TAP = 140;

  const REST = (bleed) => [`scale(${bleed})`, 1];

  class HeatmapGrid {
    constructor(root) {
      this.root = root;
      this.maps = [];

      this.geometry();
      this.bindOpacity();
      this.bindMaps();
      addEventListener("resize", () => this.geometry(), { passive: true });
    }

    bindOpacity() {
      const opacity = this.root.querySelector(".hg-opacity");
      if (!opacity) return;
      const output = this.root.querySelector(".hg-opacity-value");
      const apply = () => {
        this.root.style.setProperty("--hg-overlay-opacity", opacity.value / 100);
        if (output) output.textContent = `${opacity.value}%`;
      };
      opacity.addEventListener("input", apply);
      apply();
    }

    geometry() {
      const style = getComputedStyle(this.root);
      this.geo = {
        reach: px(style.getPropertyValue("--hg-reach"), 0.46),
        depth: px(style.getPropertyValue("--hg-depth"), 230),
        spread: px(style.getPropertyValue("--hg-spread"), 45),
        perspective: px(style.getPropertyValue("--hg-perspective"), 620),
        bleed: px(style.getPropertyValue("--hg-tile-bleed"), 1),
      };
      this.atRest = REST(this.geo.bleed);
    }

    bindMaps() {
      const toggle = this.root.querySelector(".hg-reveal-toggle");
      this.pressable = () => !calm.matches;
      this.wanted = () => (!toggle || toggle.checked) && fine.matches && this.pressable();
      if (toggle) {
        toggle.addEventListener("change", () => {
          if (!toggle.checked) this.maps.forEach((map) => this.rest(map));
        });
      }

      [...this.root.querySelectorAll(".hg-stack")].forEach((stack) => {
        if (!stack.querySelector(".hg-overlay")) return;
        const map = {
          stack,
          overlay: stack.querySelector(".hg-overlay"),
          layer: null,
          tiles: [],
          n: 0,
          live: new Set(),
          at: null,
          held: false,
          busy: false,
          origin: null,
          timer: 0,
          settle: 0,
          idle: 0,
          frame: 0,
        };
        this.maps.push(map);
        this.listen(map);
      });

      this.settleIn();
    }

    build(map) {
      clearTimeout(map.idle);
      if (map.layer) return true;
      const crisp = map.overlay.dataset.rendering !== "auto";
      const width = map.overlay.naturalWidth || 0;
      const wanted = parseInt(map.stack.dataset.tiles, 10)
        || (crisp ? width : Math.min(width, TILE_CAP));
      const n = Math.min(Math.max(wanted, 0), 64);
      if (n < 2) return false;

      const layer = document.createElement("div");
      layer.className = "hg-tiles";
      layer.style.setProperty("--hg-n", n);
      if (map.overlay.dataset.rendering) layer.dataset.rendering = map.overlay.dataset.rendering;

      const tiles = [];
      for (let i = 0; i < n * n; i++) {
        const tile = document.createElement("div");
        tile.className = "hg-tile";

        tile.style.backgroundPosition = crisp
          ? `${((i % n) / (n - 1)) * 100}% ${(((i / n) | 0) / (n - 1)) * 100}%`
          : `calc(${i % n} * (100% - 1px) / ${n - 1}) calc(${(i / n) | 0} * (100% - 1px) / ${n - 1})`;
        tiles.push(tile);
        layer.append(tile);
      }

      map.n = n;
      map.layer = layer;
      map.tiles = tiles;
      this.repoint(map);
      this.rest(map);
      map.stack.append(layer);
      this.swap(map, true);
      return true;
    }

    swap(map, tiled) {
      map.overlay.style.transition = "none";
      map.stack.classList.toggle("hg-tiled", tiled);
      void map.overlay.offsetWidth;
      map.overlay.style.transition = "";
    }

    strip(map) {
      if (!map.layer || map.held || map.busy) return;
      clearTimeout(map.idle);
      map.layer.remove();
      map.layer = null;
      map.tiles = [];
      map.n = 0;
      map.live.clear();
      this.swap(map, false);
    }

    rested(map, delay = 1500) {
      clearTimeout(map.idle);
      if (this.gone) return;
      map.idle = setTimeout(() => {
        if (map.at || map.held || map.busy) return;
        this.strip(map);
      }, delay);
    }

    repoint(map) {
      if (map.layer) map.layer.style.setProperty("--hg-map", `url("${map.overlay.src}")`);
    }

    put(tile, transform, opacity) {
      tile.style.transform = transform;
      tile.style.opacity = opacity.toFixed(3);
    }

    rest(map) {
      if (!map.layer) return;
      map.tiles.forEach((tile) => this.put(tile, this.atRest[0], this.atRest[1]));
      map.live.clear();
    }

    shrink(z) {
      const p = this.geo.perspective;
      return (this.geo.bleed * p) / (p + z);
    }

    receded(u, ux, uy) {
      const opacity = smooth(clamp01((u - HOLE) / (1 - HOLE)));
      const k = u < HOLE ? 1 - u : (1 - HOLE) * smooth(clamp01((MOVE - opacity) / MOVE));
      const out = k * this.geo.spread;
      return [
        `translate(${(ux * out).toFixed(1)}%, ${(uy * out).toFixed(1)}%)`
        + ` scale(${this.shrink(this.geo.depth * k * k).toFixed(3)})`,
        opacity,
      ];
    }

    cleared(ux, uy) {
      const out = this.geo.spread * 3.2;
      return [
        `translate(${(ux * out).toFixed(0)}%, ${(uy * out).toFixed(0)}%)`
        + ` scale(${this.shrink(this.geo.depth * PRESSED).toFixed(3)})`,
        0,
      ];
    }

    hover(map, x, y, w, h) {
      if (map.busy) return;
      const { n, tiles } = map;
      const reach = this.geo.reach * n;
      const cx = (x / w) * n - 0.5;
      const cy = (y / h) * n - 0.5;
      const next = new Set();
      const c0 = Math.max(0, Math.floor(cx - reach));
      const c1 = Math.min(n - 1, Math.ceil(cx + reach));
      const r0 = Math.max(0, Math.floor(cy - reach));
      const r1 = Math.min(n - 1, Math.ceil(cy + reach));
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c - cx;
          const dy = r - cy;
          const d = Math.hypot(dx, dy);
          if (d >= reach) continue;
          const i = r * n + c;
          const [transform, opacity] = this.receded(d / reach, dx / (d || 1), dy / (d || 1));
          this.put(tiles[i], transform, opacity);
          next.add(i);
          map.live.delete(i);
        }
      }
      map.live.forEach((i) => this.put(tiles[i], this.atRest[0], this.atRest[1]));
      map.live = next;
    }

    wave(map, out) {
      const { n, tiles, origin } = map;
      const reach = this.geo.reach * n;
      const back = !out && map.at && map.at.inside ? map.at : null;
      const far = Math.max(
        Math.hypot(origin.cx + 0.5, origin.cy + 0.5),
        Math.hypot(n - 0.5 - origin.cx, origin.cy + 0.5),
        Math.hypot(origin.cx + 0.5, n - 0.5 - origin.cy),
        Math.hypot(n - 0.5 - origin.cx, n - 0.5 - origin.cy),
      ) || 1;

      const live = new Set();
      map.layer.classList.add("hg-wave");
      for (let i = 0; i < tiles.length; i++) {
        const r = (i / n) | 0;
        const c = i % n;
        const dx = c - origin.cx;
        const dy = r - origin.cy;
        const d = Math.hypot(dx, dy);
        const norm = d / far;

        tiles[i].style.setProperty("--d", (out ? norm : 1 - norm).toFixed(3));

        let state = out ? this.cleared(dx / (d || 1), dy / (d || 1)) : this.atRest;
        if (back) {
          const bx = c - back.cx;
          const by = r - back.cy;
          const bd = Math.hypot(bx, by);

          if (bd < reach) {
            state = this.receded(bd / reach, bx / (bd || 1), by / (bd || 1));
            live.add(i);
          }
        }
        this.put(tiles[i], state[0], state[1]);
      }
      if (!out) map.live = live;
    }

    press(map, x, y, w, h) {
      clearTimeout(map.timer);

      if (map.frame) {
        cancelAnimationFrame(map.frame);
        map.frame = 0;
      }
      map.live.clear();
      map.held = true;
      map.busy = true;
      map.origin = { cx: (x / w) * map.n - 0.5, cy: (y / h) * map.n - 0.5 };
      this.wave(map, true);
    }

    release(map) {
      if (!map.held) return;
      map.held = false;
      if (!map.layer) {
        map.busy = false;
        return;
      }
      this.wave(map, false);

      map.timer = setTimeout(() => {
        map.busy = false;
        map.layer.classList.remove("hg-wave");
      }, SETTLE);
    }

    listen(map) {
      const { stack } = map;
      const local = (event) => {
        const b = stack.getBoundingClientRect();
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
        } catch (err) {  }
        const p = local(event);
        this.press(map, p.x, p.y, p.w, p.h);
      });

      ["pointerup", "pointercancel"].forEach((name) => {
        stack.addEventListener(name, (event) => {
          if (event.pointerType !== "touch") this.release(map);
        });
      });

      this.touch(stack, (x, y) => {
        if (!this.build(map)) return;
        const b = stack.getBoundingClientRect();
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

    settleIn() {
      if (calm.matches || !("IntersectionObserver" in window)) return;
      const grid = this.root.querySelector(".hg-grid");
      const cols = grid
        ? Math.max(1, getComputedStyle(grid).gridTemplateColumns.split(" ").length)
        : 1;
      this.maps.forEach((map, i) => map.stack.style.setProperty("--hg-i", i % cols));

      this.byStack = new Map(this.maps.map((map) => [map.stack, map]));
      this.seen = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          this.seen.unobserve(entry.target);
          this.arrive(this.byStack.get(entry.target));
        });
      }, { threshold: 0.4 });

      this.gone = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          const map = this.byStack.get(entry.target);
          if (!map) return;
          if (entry.isIntersecting) this.build(map);
          else this.strip(map);
        });
      }, { rootMargin: "300px" });
      this.maps.forEach((map) => {
        map.stack.classList.add("hg-armed");
        this.seen.observe(map.stack);
      });
    }

    dropped() {
      return [`scale(${this.shrink(this.geo.depth).toFixed(3)})`, 0];
    }

    arrive(map) {
      if (!map) return;
      if (!this.build(map)) {
        map.overlay.addEventListener("load", () => this.arrive(map), { once: true });
        map.settle = setTimeout(() => map.stack.classList.remove("hg-armed"), 3000);
        return;
      }

      const { stack, layer, tiles, n } = map;
      clearTimeout(map.settle);
      const far = Math.hypot(n - 1, n - 1) || 1;
      const from = this.dropped();
      layer.classList.add("hg-settling", "hg-instant");
      tiles.forEach((tile, i) => {
        tile.style.setProperty("--d", (Math.hypot(i % n, (i / n) | 0) / far).toFixed(3));
        this.put(tile, from[0], from[1]);
      });

      void layer.offsetWidth;
      layer.classList.remove("hg-instant");
      stack.classList.remove("hg-armed");
      tiles.forEach((tile) => this.put(tile, this.atRest[0], this.atRest[1]));

      const last = getComputedStyle(tiles[tiles.length - 1]);
      const ms = (parseFloat(last.transitionDelay) || 0) + (parseFloat(last.transitionDuration) || 0);
      map.settle = setTimeout(() => {
        layer.classList.remove("hg-settling");
        this.rested(map);
      }, ms * 1000 + 150);
      if (this.gone) this.gone.observe(stack);
    }
  }

  const init = (scope = document) =>
    [...scope.querySelectorAll("figure.hg[data-hg]")]
      .filter((root) => !root.dataset.hgReady)
      .map((root) => {
        root.dataset.hgReady = "true";
        return new HeatmapGrid(root);
      });

  window.HeatmapGrid = { init, Grid: HeatmapGrid };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init());
  } else {
    init();
  }
})();

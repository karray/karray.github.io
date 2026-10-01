(() => {
  "use strict";

  if (window.FigureGallery) return;

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

  const SWAP = 900;

  const HOLD = 180;
  const SLOP = 10;
  const TAP = 140;

  const REST = (bleed) => [`scale(${bleed})`, 1];

  class FigureGallery {
    constructor(root) {
      this.root = root;
      this.dock = root.querySelector(".fg-dock");
      this.thumbs = [...root.querySelectorAll(".fg-thumb")];
      this.panels = root.querySelector(".fg-panels");
      this.stacks = [...root.querySelectorAll(".fg-stack")];
      this.title = root.querySelector(".fg-title");
      this.count = root.querySelector(".fg-count");
      this.template = root.dataset.overlay || "";
      this.index = Math.max(0, this.thumbs.findIndex((t) => t.dataset.selected === "true"));
      this.pointer = null;
      this.frame = 0;
      this.maps = [];

      this.geometry();
      this.bindDock();
      this.bindControls();
      this.bindMaps();

      if (this.dock) this.select(this.index, { focus: false });
      else this.maps.forEach((map) => this.arm(map));
      addEventListener("resize", () => {
        this.geometry();
        this.measure();
      }, { passive: true });
    }

    resolve(item, stack) {
      const layer = stack.dataset.layer;
      if (!layer) return null;
      const template = item.dataset.overlay || this.template;
      if (!template) return null;
      return template
        .replace(/{layer}/g, layer)
        .replace(/{row}/g, stack.dataset.row || "")
        .replace(/{item}/g, item.dataset.item || "");
    }

    urlsFor(item) {
      const urls = [item.dataset.src];
      this.stacks.forEach((stack) => {
        const overlay = this.resolve(item, stack);
        if (overlay) urls.push(overlay);
      });
      return urls;
    }

    select(index, { focus = true } = {}) {
      const total = this.thumbs.length;
      this.index = ((index % total) + total) % total;
      const item = this.thumbs[this.index];

      this.stacks.forEach((stack) => {
        const map = this.byStack.get(stack);
        this.change(stack, map, item.dataset.src, map ? this.resolve(item, stack) : null);
      });

      this.thumbs.forEach((thumb, i) => {
        const on = i === this.index;
        thumb.setAttribute("aria-selected", on ? "true" : "false");
        thumb.tabIndex = on ? 0 : -1;
      });
      if (focus) item.focus();

      if (this.title) this.title.textContent = item.dataset.label || "";
      if (this.count) this.count.textContent = `${this.index + 1} / ${total}`;
      if (this.panels) this.panels.setAttribute("aria-labelledby", item.id);

      this.measure();
      this.paint();
      this.prefetchNeighbours();
    }

    prefetchNeighbours() {
      const idle = window.requestIdleCallback
        ? window.requestIdleCallback.bind(window)
        : (fn) => setTimeout(fn, 400);
      idle(() => {
        [-1, 1].forEach((step) => {
          const total = this.thumbs.length;
          const neighbour = this.thumbs[(this.index + step + total) % total];
          if (neighbour && neighbour !== this.thumbs[this.index]) {
            this.urlsFor(neighbour).forEach((url) => {
              const img = new Image();
              img.decoding = "async";
              img.src = url;
            });
          }
        });
      });
    }

    metrics() {
      const style = getComputedStyle(this.root);
      return this.fit({
        min: px(style.getPropertyValue("--fg-thumb-min"), 40),
        selected: px(style.getPropertyValue("--fg-thumb-selected"), 62),
        max: px(style.getPropertyValue("--fg-thumb-max"), 80),
        range: px(style.getPropertyValue("--fg-magnify-range"), 110),
        gap: px(getComputedStyle(this.dock).columnGap, 8),
      });
    }

    fit(m) {
      const gaps = m.gap * Math.max(0, this.thumbs.length - 1);
      const rest = this.thumbs.reduce((sum, _, i) => sum + this.restSize(i, m), 0);
      const available = this.dock.clientWidth;
      const wanted = rest + gaps + 2.4 * (m.max - m.min);
      if (available > 0 && wanted > available) {
        const k = Math.max(0.4, (available - gaps) / (wanted - gaps));
        return { ...m, min: m.min * k, selected: m.selected * k, max: m.max * k, range: m.range * k };
      }
      return m;
    }

    restSize(i, m) {
      const falloff = Math.max(0, 1 - Math.abs(i - this.index) / 3) ** 2;
      return m.min + (m.selected - m.min) * falloff;
    }

    measure() {
      if (!this.dock) return;
      const m = this.metrics();
      const sizes = this.thumbs.map((_, i) => this.restSize(i, m));
      const width = sizes.reduce((a, b) => a + b, 0) + m.gap * Math.max(0, sizes.length - 1);
      let x = Math.max(0, (this.dock.clientWidth - width) / 2);
      this.centres = sizes.map((size) => {
        const centre = x + size / 2;
        x += size + m.gap;
        return centre;
      });
      this.m = m;
    }

    paint() {
      const m = this.m || this.metrics();
      const magnify = this.pointer !== null && fine.matches && !calm.matches;
      this.thumbs.forEach((thumb, i) => {
        let size = this.restSize(i, m);
        if (magnify) {
          const t = Math.abs(this.pointer - this.centres[i]) / m.range;
          if (t < 1) {
            const bump = Math.cos((t * Math.PI) / 2) ** 2;
            size = Math.max(size, m.min + (m.max - m.min) * bump);
          }
        }
        thumb.style.setProperty("--fg-size", `${Math.round(size)}px`);
      });
    }

    schedule() {
      if (this.frame) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        this.paint();
      });
    }

    bindDock() {
      if (!this.dock) return;
      this.dock.addEventListener("pointermove", (event) => {
        if (event.pointerType !== "mouse") return;
        const bounds = this.dock.getBoundingClientRect();
        this.pointer = event.clientX - bounds.left + this.dock.scrollLeft;
        this.schedule();
      });
      this.dock.addEventListener("pointerleave", () => {
        this.pointer = null;
        this.schedule();
      });

      this.thumbs.forEach((thumb, i) => {
        thumb.addEventListener("click", () => this.select(i, { focus: false }));
      });

      this.dock.addEventListener("keydown", (event) => {
        const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
        if (step) this.select(this.index + step);
        else if (event.key === "Home") this.select(0);
        else if (event.key === "End") this.select(this.thumbs.length - 1);
        else return;
        event.preventDefault();
        this.thumbs[this.index].scrollIntoView({ block: "nearest", inline: "nearest" });
      });
    }

    bindControls() {
      const opacity = this.root.querySelector(".fg-opacity");
      if (opacity) {
        const output = this.root.querySelector(".fg-opacity-value");
        const apply = () => {
          this.root.style.setProperty("--fg-overlay-opacity", opacity.value / 100);
          if (output) output.textContent = `${opacity.value}%`;
        };
        opacity.addEventListener("input", apply);
        apply();
      }

      this.root.querySelectorAll("[data-axis-target]").forEach((el) => {
        const target = el.dataset.axisTarget;
        const output = el.closest("label")?.querySelector(".fg-axis-value");
        const event = el.tagName === "SELECT" ? "change" : "input";
        el.addEventListener(event, () => {
          this.stacks.forEach((stack) => { stack.dataset[target] = el.value; });
          if (output) output.textContent = el.value;
          this.restage();
        });
      });
    }

    restage() {
      const item = this.thumbs[this.index];
      this.maps.forEach((map) => {
        const src = this.resolve(item, map.stack);
        if (src) this.change(map.stack, map, null, src);
      });
    }

    change(stack, map, baseSrc, overlaySrc) {
      const base = stack.querySelector(".fg-base");
      const overlay = stack.querySelector(".fg-overlay");
      const setBase = () => { if (base && baseSrc) base.src = baseSrc; };

      if (!map || !map.arrived) {
        setBase();
        if (overlay && overlaySrc) overlay.src = overlaySrc;
        if (map) {
          this.repoint(map);
          this.arm(map);
        }
        return;
      }

      clearTimeout(map.timer);
      if (map.frame) {
        cancelAnimationFrame(map.frame);
        map.frame = 0;
      }
      map.held = false;
      map.busy = false;

      const ghost = this.ghost(stack, map);
      const pending = [];
      setBase();
      if (base && baseSrc) pending.push(base);
      if (overlay && overlaySrc) {
        overlay.src = overlaySrc;
        pending.push(overlay);
      }
      this.ready(pending, () => {
        this.repoint(map);

        if (!this.build(map)) this.swap(map, false);

        ghost.classList.add("fg-going");
        setTimeout(() => ghost.remove(), SWAP);
      });
    }

    ghost(stack, map) {
      const stale = stack.querySelector(".fg-ghost");
      if (stale) stale.remove();
      const ghost = document.createElement("div");
      ghost.className = "fg-ghost";
      const base = stack.querySelector(".fg-base");
      if (base && base.getAttribute("src")) {
        const copy = base.cloneNode();
        copy.className = "fg-ghost-base";
        ghost.append(copy);
      }
      if (map.layer) {
        ghost.append(map.layer);
        map.layer = null;
        map.tiles = [];
        map.n = 0;
        map.live.clear();
      } else if (map.overlay.getAttribute("src")) {
        const copy = map.overlay.cloneNode();
        copy.className = "fg-ghost-map";
        ghost.append(copy);
      }
      stack.append(ghost);
      return ghost;
    }

    ready(imgs, done) {
      let left = imgs.length;
      if (!left) return done();
      const tick = () => { if (!--left) done(); };
      imgs.forEach((img) => {
        if (img.complete && img.naturalWidth) return tick();
        img.addEventListener("load", tick, { once: true });
        img.addEventListener("error", tick, { once: true });
      });
    }

    geometry() {
      const style = getComputedStyle(this.root);
      this.geo = {
        reach: px(style.getPropertyValue("--fg-reach"), 0.46),
        depth: px(style.getPropertyValue("--fg-depth"), 230),
        spread: px(style.getPropertyValue("--fg-spread"), 45),
        perspective: px(style.getPropertyValue("--fg-perspective"), 620),
        bleed: px(style.getPropertyValue("--fg-tile-bleed"), 1),
      };
      this.atRest = REST(this.geo.bleed);
    }

    bindMaps() {
      const toggle = this.root.querySelector(".fg-reveal-toggle");
      this.pressable = () => !calm.matches;
      this.wanted = () => (!toggle || toggle.checked) && fine.matches && this.pressable();
      if (toggle) {
        toggle.addEventListener("change", () => {
          if (!toggle.checked) this.maps.forEach((map) => this.rest(map));
        });
      }

      this.stacks.forEach((stack) => {
        if (!stack.querySelector(".fg-overlay")) return;
        const map = {
          stack,
          overlay: stack.querySelector(".fg-overlay"),
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
      });
      this.byStack = new Map(this.maps.map((map) => [map.stack, map]));

      if (this.root.dataset.reveal === "continuous") this.sheet();
      else this.maps.forEach((map) => this.listen(map));

      this.settleIn();
    }

    sheet() {
      const area = this.root.querySelector(".fg-grid") || this.root;
      let frame = 0;
      let at = null;

      const each = (fn) => this.maps.forEach((map) => {
        if (!this.build(map)) return;
        const b = map.stack.getBoundingClientRect();
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
        } catch (err) {  }
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
      layer.className = "fg-tiles";
      layer.style.setProperty("--fg-n", n);
      if (map.overlay.dataset.rendering) layer.dataset.rendering = map.overlay.dataset.rendering;

      const tiles = [];
      for (let i = 0; i < n * n; i++) {
        const tile = document.createElement("div");
        tile.className = "fg-tile";

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
      map.stack.classList.toggle("fg-tiled", tiled);
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
      if (map.layer) map.layer.style.setProperty("--fg-map", `url("${map.overlay.src}")`);
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
      map.layer.classList.add("fg-wave");
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
        map.layer.classList.remove("fg-wave");
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

    settleIn() {
      if (calm.matches || !("IntersectionObserver" in window)) return;
      const cols = Math.max(1, parseInt(getComputedStyle(this.root).getPropertyValue("--fg-cols"), 10) || 1);

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
      this.maps.forEach((map, i) => map.stack.style.setProperty("--fg-i", i % cols));
    }

    arm(map) {
      if (!this.seen) return;
      clearTimeout(map.settle);
      clearTimeout(map.idle);
      this.seen.unobserve(map.stack);
      map.stack.classList.add("fg-armed");
      map.live.clear();
      if (map.layer) map.layer.classList.remove("fg-wave", "fg-settling");
      this.seen.observe(map.stack);
    }

    dropped() {
      return [`scale(${this.shrink(this.geo.depth).toFixed(3)})`, 0];
    }

    arrive(map) {
      if (!map) return;
      if (!this.build(map)) {
        map.overlay.addEventListener("load", () => this.arrive(map), { once: true });
        map.settle = setTimeout(() => map.stack.classList.remove("fg-armed"), 3000);
        return;
      }

      const { stack, layer, tiles, n } = map;
      clearTimeout(map.settle);
      const far = Math.hypot(n - 1, n - 1) || 1;
      const from = this.dropped();
      layer.classList.remove("fg-wave");
      layer.classList.add("fg-settling", "fg-instant");
      tiles.forEach((tile, i) => {
        tile.style.setProperty("--d", (Math.hypot(i % n, (i / n) | 0) / far).toFixed(3));
        this.put(tile, from[0], from[1]);
      });

      void layer.offsetWidth;
      layer.classList.remove("fg-instant");
      stack.classList.remove("fg-armed");
      tiles.forEach((tile) => this.put(tile, this.atRest[0], this.atRest[1]));

      map.arrived = true;
      const last = getComputedStyle(tiles[tiles.length - 1]);
      const ms = (parseFloat(last.transitionDelay) || 0) + (parseFloat(last.transitionDuration) || 0);
      map.settle = setTimeout(() => {
        layer.classList.remove("fg-settling");
        this.rested(map);
      }, ms * 1000 + 150);
      if (this.gone) this.gone.observe(stack);
    }
  }

  const init = (scope = document) =>
    [...scope.querySelectorAll("figure.fg[data-fg]")]
      .filter((root) => !root.dataset.fgReady)
      .map((root) => {
        root.dataset.fgReady = "true";
        return new FigureGallery(root);
      });

  window.FigureGallery = { init, Gallery: FigureGallery };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init());
  } else {
    init();
  }
})();

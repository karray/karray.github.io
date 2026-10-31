(() => {
  "use strict";

  if (window.FigureGallery) return;

  const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
  const px = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const SWAP = 900;

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
      this.notes = [...root.querySelectorAll(".fg-cell-note")];
      try { this.noteText = JSON.parse(root.dataset.notes || "{}"); } catch { this.noteText = {}; }

      this.bindDock();
      this.bindControls();
      this.tiles = new window.TileReveal(root, {
        prefix: "fg",
        autoArm: false,
        getCols: () => Math.max(1, parseInt(getComputedStyle(root).getPropertyValue("--fg-cols"), 10) || 1),
      });
      this.maps = this.tiles.maps;
      this.byStack = this.tiles.byStack;

      if (this.dock) this.select(this.index, { focus: false });
      else this.tiles.armAll();

      addEventListener("resize", () => this.measure(), { passive: true });
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

      this.annotate();
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
        let labels = null;
        try { labels = el.dataset.labels ? JSON.parse(el.dataset.labels) : null; } catch { labels = null; }
        el.addEventListener(event, () => {
          this.stacks.forEach((stack) => { stack.dataset[target] = el.value; });
          if (output) output.textContent = labels ? labels[Number(el.value)] ?? el.value : el.value;
          this.restage();
          this.annotate();
        });
      });
    }

    annotate() {
      if (!this.notes.length) return;
      const item = this.thumbs[this.index];
      this.notes.forEach((note, i) => {
        const stack = this.stacks[i];
        const key = stack ? `${stack.dataset.row || ""}|${stack.dataset.layer || ""}|${item?.dataset.item || ""}` : "";
        note.textContent = this.noteText[key] || "";
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
        if (map) this.tiles.arm(map);
        return;
      }

      this.tiles.cancel(map);

      const ghost = this.ghost(stack, map);
      const pending = [];
      setBase();
      if (base && baseSrc) pending.push(base);
      if (overlay && overlaySrc) {
        overlay.src = overlaySrc;
        pending.push(overlay);
      }
      this.ready(pending, () => {
        if (map.at && map.at.inside) {
          if (!this.tiles.build(map)) this.tiles.swap(map, false);
        } else {
          this.tiles.strip(map);
        }
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
      const canvas = this.tiles.detachCanvas(map);
      if (canvas) {
        ghost.append(canvas);
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

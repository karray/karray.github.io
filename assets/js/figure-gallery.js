/* Reusable figure gallery.
 *
 * Drives every <figure class="fg" data-fg> on the page. The markup is rendered
 * server-side for the first item (so the figure is complete without JS); this
 * script only swaps image sources, magnifies the dock, runs the reveal
 * gesture and settles the maps in. Nothing here is specific to a single post.
 *
 * Markup contract:
 *   figure.fg[data-fg]            data-overlay="/path/{layer}_{row}_{item}.png"
 *   button.fg-thumb              data-item, data-src, optional data-overlay
 *   .fg-stack                    data-row, optional data-layer, data-tiles
 *   .fg-stack img.fg-base        the photograph
 *   .fg-stack img.fg-overlay     the map drawn on top
 */
(() => {
  "use strict";

  const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
  const px = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  };
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  /* flat at both ends, so the reveal has no visible rim */
  const smooth = (t) => t * t * (3 - 2 * t);

  /* a 7px map becomes 7 tiles and a 14px map 14; anything smoother is
     tiled at the cap, which is plenty for a gesture */
  const TILE_CAP = 14;
  /* the inner share of the reach that is cleared completely */
  const HOLE = 0.45;
  /* how much further than the pointer's own depth a pressed map falls */
  const PRESSED = 2.4;
  /* long enough for the wave to cross the map and for every tile to land */
  const SETTLE = 900;

  const REST = ["scale(1.04)", 1];

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
      this.select(this.index, { focus: false });
      addEventListener("resize", () => {
        this.geometry();
        this.measure();
      }, { passive: true });
    }

    /* ----------------------------------------------------------- sources */

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
        const base = stack.querySelector(".fg-base");
        const overlay = stack.querySelector(".fg-overlay");
        if (base) base.src = item.dataset.src;
        if (overlay) {
          const src = this.resolve(item, stack);
          if (src) overlay.src = src;
        }
      });
      this.maps.forEach((map) => {
        this.repoint(map);
        this.arm(map);
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
      /* bound: a detached native call would throw under strict mode */
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

    /* -------------------------------------------------------------- dock */

    /* Rest sizes fall off with the distance to the selected item; the cursor
       adds a macOS-dock style bump on top. Both are measured in px so a post
       can retune the dock purely through the CSS custom properties. */
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

    /* The dock must never scroll or spill, so when the row - plus the width
       the cursor bump and its overshoot will add to it - does not fit,
       shrink the whole dock proportionally instead. */
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

    /* Item centres of the *rest* layout: magnifying must not move the
       reference points, or the sizes would feed back into themselves. */
    measure() {
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

    /* ---------------------------------------------------------- controls */

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
    }

    /* ------------------------------------------------------------- maps */

    geometry() {
      const style = getComputedStyle(this.root);
      this.geo = {
        reach: px(style.getPropertyValue("--fg-reach"), 0.46),
        depth: px(style.getPropertyValue("--fg-depth"), 230),
        spread: px(style.getPropertyValue("--fg-spread"), 45),
        perspective: px(style.getPropertyValue("--fg-perspective"), 620),
      };
    }

    /* Reveal gesture. The map is cut into tiles that fall back through the
       panel's perspective and fade out around the pointer, so the
       photograph underneath shows through. Pressing sends the same effect
       across the whole map as a wave from the point pressed; releasing
       plays that wave backwards, into whatever the pointer is doing by
       then. Tiles are built on first use, so twenty panels cost nothing
       until one of them is actually touched. */
    bindMaps() {
      const toggle = this.root.querySelector(".fg-reveal-toggle");
      this.wanted = () => (!toggle || toggle.checked) && fine.matches && !calm.matches;
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
          held: false,   /* the pointer is down */
          busy: false,   /* a press or return wave is still in flight */
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
      const wanted = parseInt(map.stack.dataset.tiles, 10)
        || Math.min(map.overlay.naturalWidth || 0, TILE_CAP);
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
        /* one cell of the map per tile, cut straight out of the PNG */
        tile.style.backgroundPosition =
          `${((i % n) / (n - 1)) * 100}% ${(((i / n) | 0) / (n - 1)) * 100}%`;
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

    /* The image and the tile layer show the same map, so handing over from
       one to the other has to be instant: the image carries a fade for the
       opacity slider, and through it the photograph would show bare. */
    swap(map, tiled) {
      map.overlay.style.transition = "none";
      map.stack.classList.toggle("fg-tiled", tiled);
      void map.overlay.offsetWidth;
      map.overlay.style.transition = "";
    }

    /* A map is worth cutting into tiles only while something is happening
       to it. Left in place they are the figure's whole cost: a hundred or
       two paint chunks per panel, twenty panels deep, for a page that is
       showing plain images the rest of the time. */
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

    /* the pointer often comes straight back, so wait before tearing down */
    rested(map, delay = 1500) {
      clearTimeout(map.idle);
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
      map.tiles.forEach((tile) => this.put(tile, REST[0], REST[1]));
      map.live.clear();
    }

    /* How much a tile at depth `z` shrinks. A tile is projected around its
       own centre, so the perspective divide is just a scale - written as
       one, the transform stays 2D and the tile never forces a 3D
       rendering context or the compositing that comes with it. */
    shrink(z) {
      const p = this.geo.perspective;
      return (1.04 * p) / (p + z);
    }

    /* Recede. `u` is how far a tile sits from the pointer as a share of the
       reach and (ux, uy) points from the pointer towards the tile: the
       inner HOLE of the reach is cleared completely and the rest is a
       smoothstep, so the opening has no rim.

       The tile falls back and slides *away* from the pointer, never
       towards it. Depth runs as the square of the ramp, which keeps a tile
       that is still visible close to its resting size - otherwise it
       shrinks and gaps open between tiles that have not faded yet. */
    receded(u, ux, uy) {
      const k = 1 - u;
      const out = k * this.geo.spread;
      return [
        `translate(${(ux * out).toFixed(1)}%, ${(uy * out).toFixed(1)}%)`
        + ` scale(${this.shrink(this.geo.depth * k * k).toFixed(3)})`,
        smooth(clamp01((u - HOLE) / (1 - HOLE))),
      ];
    }

    /* the full-strength state a press throws the whole map into */
    cleared(ux, uy) {
      const out = this.geo.spread * 3.2;
      return [
        `translate(${(ux * out).toFixed(0)}%, ${(uy * out).toFixed(0)}%)`
        + ` scale(${this.shrink(this.geo.depth * PRESSED).toFixed(3)})`,
        0,
      ];
    }

    /* Only the tiles inside the reach are written, plus the ones that have
       just left it: a frame is a few dozen style writes, not n squared. */
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
      map.live.forEach((i) => this.put(tiles[i], REST[0], REST[1]));
      map.live = next;
    }

    /* Press and release: one pass over the grid sets a normalised delay per
       tile, and the browser runs the whole wave from that alone. */
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
        /* going out, the nearest tiles leave first; coming back, the wave
           runs in reverse, so the ones that left last return first */
        tiles[i].style.setProperty("--d", (out ? norm : 1 - norm).toFixed(3));

        let state = out ? this.cleared(dx / (d || 1), dy / (d || 1)) : REST;
        if (back) {
          const bx = c - back.cx;
          const by = r - back.cy;
          const bd = Math.hypot(bx, by);
          /* the map must come back to whatever the pointer is doing now,
             not to a flat resting state it would then have to undo */
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
      /* a mouse always sends a move just before the button goes down, and
         that move's frame would otherwise land on top of the press wave */
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
      this.wave(map, false);
      /* stay busy until the return wave has landed, so a stray pointermove
         cannot fight it, then drop the stagger for a snappy hover again */
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
        /* recorded even while a wave is running: the release needs to know
           where the pointer ended up */
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

      stack.addEventListener("pointerleave", () => {
        map.at = null;
        this.release(map);
        /* also while a wave is in flight: the release targets whatever the
           pointer was doing, and the pointer has just gone */
        if (!map.held) this.rest(map);
        this.rested(map);
      });

      stack.addEventListener("pointerdown", (event) => {
        if (event.pointerType === "touch" || event.button !== 0) return;
        if (!this.wanted() || !this.build(map)) return;
        /* a press is a gesture, not the start of a text selection */
        event.preventDefault();
        /* keeps the release ours even if the pointer leaves the panel;
           a pointer that is no longer active refuses to be captured */
        try {
          stack.setPointerCapture(event.pointerId);
        } catch (err) { /* not capturable - release still arrives here */ }
        const p = local(event);
        this.press(map, p.x, p.y, p.w, p.h);
      });

      ["pointerup", "pointercancel"].forEach((name) => {
        stack.addEventListener(name, () => this.release(map));
      });
    }

    /* Maps are held back until their own panel scrolls into view, then
       settle in a column at a time. The arming itself happens in select(),
       so switching image replays the arrival instead of snapping the new
       maps into place. */
    settleIn() {
      if (calm.matches || !("IntersectionObserver" in window)) return;
      const cols = Math.max(1, parseInt(getComputedStyle(this.root).getPropertyValue("--fg-cols"), 10) || 1);
      /* once a map has arrived the classes have to go, or the filled
         animation would outrank the tile layer hiding the image below it */
      this.byStack = new Map(this.maps.map((map) => [map.stack, map]));
      this.seen = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          this.seen.unobserve(entry.target);
          this.arrive(this.byStack.get(entry.target));
        });
      }, { threshold: 0.4 });
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

    /* the state a tile is assembled from: deep behind the panel, invisible */
    dropped() {
      return [`scale(${this.shrink(this.geo.depth).toFixed(3)})`, 0];
    }

    /* The map builds itself out of its own tiles, in a wave from the top
       left corner. The end is timed off the tiles' own CSS rather than
       listened for: a missed transitionend - a hidden tab, an interrupted
       transition - would leave the map invisible for good. */
    arrive(map) {
      if (!map) return;
      if (!this.build(map)) {
        /* nothing measurable to cut up yet: try again when the image lands,
           and never leave the panel blank if it never does */
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

      /* the start state has to become a style of its own, or the browser
         would only ever see the resting one and nothing would move */
      void layer.offsetWidth;
      layer.classList.remove("fg-instant");
      stack.classList.remove("fg-armed");
      tiles.forEach((tile) => this.put(tile, REST[0], REST[1]));

      const last = getComputedStyle(tiles[tiles.length - 1]);
      const ms = (parseFloat(last.transitionDelay) || 0) + (parseFloat(last.transitionDuration) || 0);
      map.settle = setTimeout(() => {
        layer.classList.remove("fg-settling");
        this.rested(map);
      }, ms * 1000 + 150);
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

/* Reusable heatmap grid: a fixed set of independent image+heatmap cards,
 * each with its own reveal gesture, laid out as a plain grid - no dock, no
 * row/column axes (see figure-gallery.js for that variant). Drives every
 * <figure class="hg" data-hg> on the page: the shared opacity slider, the
 * reveal-tile gesture per card, and the scroll-in arrival animation.
 *
 * Markup contract:
 *   figure.hg[data-hg]
 *   .hg-stack                  optional data-tiles
 *   .hg-stack img.hg-base      the photograph
 *   .hg-stack img.hg-overlay   the heatmap drawn on top
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

  /* Every tile state carries the bleed (--hg-tile-bleed): tiles overlap by
     that much at rest and all through the gesture, so a subpixel seam can
     never open between two of them. The overlay image carries the same
     scale in CSS - the map has to be the same size whether it is being
     drawn as one image or as a grid of tiles, or handing over between the
     two on hover would resize it. */
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

    /* ---------------------------------------------------------- controls */

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

    /* ------------------------------------------------------------- maps */

    geometry() {
      const style = getComputedStyle(this.root);
      this.geo = {
        reach: px(style.getPropertyValue("--hg-reach"), 0.46),
        depth: px(style.getPropertyValue("--hg-depth"), 230),
        spread: px(style.getPropertyValue("--hg-spread"), 45),
        perspective: px(style.getPropertyValue("--hg-perspective"), 620),
        bleed: px(style.getPropertyValue("--hg-tile-bleed"), 1.04),
      };
      this.atRest = REST(this.geo.bleed);
    }

    /* Reveal gesture. The map is cut into tiles that fall back through the
       card's perspective and fade out around the pointer, so the
       photograph underneath shows through. Pressing sends the same effect
       across the whole map as a wave from the point pressed; releasing
       plays that wave backwards, into whatever the pointer is doing by
       then. Tiles are built on first use, so a grid of many cards costs
       nothing until one of them is actually touched. */
    bindMaps() {
      const toggle = this.root.querySelector(".hg-reveal-toggle");
      this.wanted = () => (!toggle || toggle.checked) && fine.matches && !calm.matches;
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
      layer.className = "hg-tiles";
      layer.style.setProperty("--hg-n", n);
      if (map.overlay.dataset.rendering) layer.dataset.rendering = map.overlay.dataset.rendering;

      const tiles = [];
      for (let i = 0; i < n * n; i++) {
        const tile = document.createElement("div");
        tile.className = "hg-tile";
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
      map.stack.classList.toggle("hg-tiled", tiled);
      void map.overlay.offsetWidth;
      map.overlay.style.transition = "";
    }

    /* A map is worth cutting into tiles only while something is happening
       to it. Left in place they are the grid's whole cost: a hundred or
       two paint chunks per card, dozens of cards deep, for a page that is
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

    /* Once a map has arrived, the tile layer is what the reader is looking
       at, and it stays that way for as long as the panel is on screen.
       Handing back to the overlay image whenever the pointer rests would
       move every cell edge by the bleed and back again - the image has no
       way to reproduce a per-tile overlap - so the swap is worth having
       exactly once, off screen, where `gone` does it. */
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

    /* How much a tile at depth `z` shrinks. A tile is projected around its
       own centre, so the perspective divide is just a scale - written as
       one, the transform stays 2D and the tile never forces a 3D
       rendering context or the compositing that comes with it. */
    shrink(z) {
      const p = this.geo.perspective;
      return (this.geo.bleed * p) / (p + z);
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
      map.live.forEach((i) => this.put(tiles[i], this.atRest[0], this.atRest[1]));
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
      map.layer.classList.add("hg-wave");
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

        let state = out ? this.cleared(dx / (d || 1), dy / (d || 1)) : this.atRest;
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
        /* keeps the release ours even if the pointer leaves the card;
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

    /* Cards are held back until they scroll into view, then settle in a
       row at a time: --hg-i is each card's column within the grid's own
       track count, measured after layout, so cards that enter together
       fall in a left-to-right wave instead of all at once. */
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

      /* the panels the reader has scrolled past are where the tile layer's
         cost is worth reclaiming: a hundred or two paint chunks apiece,
         many panels deep, for maps nobody is looking at. Coming back it is
         rebuilt silently, and far enough ahead of the panel being on screen
         that the reader never catches the swap - a panel in view is always
         showing its tiles, so hovering one never hands over. */
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

    /* the state a tile is assembled from: deep behind the card, invisible */
    dropped() {
      return [`scale(${this.shrink(this.geo.depth).toFixed(3)})`, 0];
    }

    /* The map builds itself out of its own tiles, in a wave from the top
       left corner. The end is timed off the tiles' own CSS rather than
       listened for: a missed transitionend - a hidden tab, an interrupted
       transition - would leave the card invisible for good. */
    arrive(map) {
      if (!map) return;
      if (!this.build(map)) {
        /* nothing measurable to cut up yet: try again when the image lands,
           and never leave the card blank if it never does */
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

      /* the start state has to become a style of its own, or the browser
         would only ever see the resting one and nothing would move */
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

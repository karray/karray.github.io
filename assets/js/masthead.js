(() => {
  const hero = document.querySelector(".hd"), bar_ = document.querySelector(".hd-bar");
  if (!hero) return;
  const root = document.documentElement, both = (f) => { f(hero); if (bar_) f(bar_); }, canvas = hero.querySelector("canvas.fx"), ctx = canvas && canvas.getContext("2d");
  const BAYER = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21].map((v) => (v + 0.5) / 64);
  // back to front: height (fraction of the header), amplitude (px), frequency, phase
  const RIDGES = [[0.32, 14, 0.00952, 4.58], [0.47, 23.6, 0.00598, 3.41], [0.62, 26.3, 0.00782, 2.35], [0.77, 26.7, 0.00819, 2.94], [0.92, 30.6, 0.00884, 5.8]];
  const CELL = 3;
  const ss = (a, b, x) => { x = Math.min(1, Math.max(0, (x - a) / (b - a))); return x * x * (3 - 2 * x); };
  const rgb = (s) => { const n = parseInt(s.trim().slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let cols, rows, cp, W, H, bar, colSide, colour, crest, img, p = 0, lastY = scrollY, travel = 0;
  // the warp follows the pointer (tx, ty) and eases to strength ta
  const ptr = { x: 0, y: 0, tx: 0, ty: 0, a: 0, ta: 0 }, AMP = 24, SIGMA = 70, GLOW = 0.45;
  let raf = 0;

  function size() {
    bar = parseFloat(getComputedStyle(root).getPropertyValue("--bar")); H = hero.clientHeight;
    // a header that fits its text (a post's title can wrap) tells the CSS its height
    if (hero.classList.contains("fit")) hero.style.setProperty("--h", H + "px");
    if (!canvas) return;
    const dpr = devicePixelRatio || 1;
    cp = Math.max(1, Math.round(CELL * dpr)) / dpr;
    W = hero.clientWidth;
    cols = Math.ceil(W / cp); rows = Math.ceil(H / cp);
    canvas.width = cols; canvas.height = rows;
    canvas.style.width = cols * cp + "px"; canvas.style.height = rows * cp + "px";
    img = ctx.createImageData(cols, rows);
    // narrower headers get narrower dunes, so a phone still shows whole ones; in the middle, about the
    // text column wide on a wide screen and 60% of a narrow one, the dunes turn grey and only their crests remain
    const sx = Math.sqrt(1440 / W), grey = Math.min(290, W * 0.3);
    colSide = new Float32Array(cols); crest = new Float32Array(RIDGES.length * cols);
    for (let x = 0; x < cols; x++) {
      const px = (x + 0.5) * cp, qx = px * sx;
      colSide[x] = ss(grey, grey + Math.min(330, W * 0.2), Math.abs(px - W / 2));
      RIDGES.forEach(([base, amp, f, ph], k) => (crest[k * cols + x] = H * base - amp * (1 + Math.sin(qx * f + ph)) - 10 * Math.sin(qx * f * 2.7 + ph)));
    }
    colours();
  }
  function colours() {
    if (!canvas) return;
    const cs = getComputedStyle(canvas), a = rgb(cs.getPropertyValue("--fx-ink")), z = rgb(cs.getPropertyValue("--fx-ink2")), g = rgb(cs.getPropertyValue("--fx-mid"));
    colour = new Uint8ClampedArray(cols * 3);
    for (let x = 0; x < cols; x++) {
      const px = (x + 0.5) * cp, t = Math.min(1, Math.min(px, W - px) / (W * 0.25)), m = 1 - colSide[x];
      for (let c = 0; c < 3; c++) { const e = a[c] + (z[c] - a[c]) * t; colour[x * 3 + c] = e + (g[c] - e) * m; }
    }
  }
  function reach() {
    const R = ptr.a > 0.005 ? 3.4 * SIGMA : -Infinity;
    return { x0: ptr.x - R, x1: ptr.x + R, y0: ptr.y - R, y1: ptr.y + R };
  }
  function draw(patch) {
    if (!canvas) return;
    const { x0, x1, y0, y1 } = reach(), S2 = 2 * SIGMA * SIGMA, warp = ptr.a;
    const d = img.data, fade = (1 - p) ** 1.3, shift = Math.round((p * (H - bar)) / 2 / cp);
    const c0 = patch ? Math.max(0, Math.floor(patch.x0 / cp)) : 0, c1 = patch ? Math.min(cols, Math.ceil(patch.x1 / cp)) : cols;
    const r0 = patch ? Math.max(0, Math.floor(patch.y0 / cp)) : 0, r1 = patch ? Math.min(rows, Math.ceil(patch.y1 / cp)) : rows;
    if (c0 >= c1 || r0 >= r1) return;
    for (let yy = r0; yy < r1; yy++) d.fill(0, (yy * cols + c0) * 4, (yy * cols + c1) * 4);
    if (fade > 0.01)
      for (let y = r0 - shift; y < r1 - shift; y++) {
        if (y < 0) continue;
        const py = (y + 0.5) * cp, rs = fade, yy = y + shift;
        if (rs <= 0 || yy >= rows) continue;
        for (let x = c0; x < c1; x++) {
          let glow = 0;
          const cx = (x + 0.5) * cp, cy = (yy + 0.5) * cp;
          if (cx > x0 && cx < x1 && cy > y0 && cy < y1) { const dx = cx - ptr.x, dy = cy - ptr.y; glow = warp * Math.exp(-(dx * dx + dy * dy) / S2); }
          const qy = py + AMP * glow;
          let v = 0;
          for (let k = RIDGES.length - 1; k >= 0; k--) {
            const c = crest[k * cols + x];
            // the body, thinning out toward the middle, and the crest, a solid line 3px thick everywhere: a
            // thinned-out crest would break into the dither's sparse grid
            if (qy >= c - 3) { v = Math.max((0.55 + 0.12 * k) * ss(c - 3, c + 2, qy) * (1 - 0.82 * ss(c, c + 46, qy)) * colSide[x] ** 1.4, ss(c - 3, c - 1, qy) * (1 - ss(c + 2, c + 4, qy))); break; }
          }
          if (v * (1 + GLOW * glow) * rs <= BAYER[(yy & 7) * 8 + (x & 7)]) continue;
          const i = (yy * cols + x) * 4, c = x * 3;
          d[i] = colour[c]; d[i + 1] = colour[c + 1]; d[i + 2] = colour[c + 2]; d[i + 3] = 255;
        }
      }
    if (patch) ctx.putImageData(img, 0, 0, c0, r0, c1 - c0, r1 - r0);
    else ctx.putImageData(img, 0, 0);
  }
  let last = 0, before = null;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (now - last < 30) return;
    last = now;
    ptr.x += (ptr.tx - ptr.x) * 0.3; ptr.y += (ptr.ty - ptr.y) * 0.3;
    ptr.a += (ptr.ta - ptr.a) * 0.18;
    const done = Math.hypot(ptr.tx - ptr.x, ptr.ty - ptr.y) < 0.3 && Math.abs(ptr.ta - ptr.a) < 0.005;
    if (done) { ptr.x = ptr.tx; ptr.y = ptr.ty; ptr.a = ptr.ta; }
    // redraw what the warp covers now and covered last frame
    const now_ = reach(), patch = before ? { x0: Math.min(before.x0, now_.x0), x1: Math.max(before.x1, now_.x1), y0: Math.min(before.y0, now_.y0), y1: Math.max(before.y1, now_.y1) } : now_;
    before = done && !ptr.a ? null : now_;
    draw(Number.isFinite(patch.x0) ? patch : undefined);
    if (done) { cancelAnimationFrame(raf); raf = 0; }
  }
  function move(e) {
    const b = hero.getBoundingClientRect();
    ptr.tx = e.clientX - b.left; ptr.ty = e.clientY - b.top;
    if (ptr.a < 0.01) { ptr.x = ptr.tx; ptr.y = ptr.ty; }
    ptr.ta = p >= 1 ? 0 : 1;
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function leave() { ptr.ta = 0; if (!raf) raf = requestAnimationFrame(frame); }

  function onScroll() {
    const y = scrollY, step = y - lastY, was = p;
    lastY = y;
    p = Math.min(1, Math.max(0, y / (H - bar)));
    both((el) => el.classList.toggle("is-docked", p >= 1));
    if (Math.abs(step) >= innerHeight) travel = 0;
    else if (step) travel = Math.sign(step) === Math.sign(travel) ? travel + step : step;
    // travel counts from the docking point, so short pages hide too
    if (p < 1) { travel = 0; both((el) => el.classList.remove("is-hidden")); }
    else if (Math.abs(travel) >= 48) both((el) => el.classList.toggle("is-hidden", travel > 0));
    if (p !== was) draw();
  }
  const scroll = () => requestAnimationFrame(onScroll);
  const resize = () => { size(); draw(); };
  const theme = new MutationObserver(() => { colours(); draw(); });
  theme.observe(root, { attributes: true, attributeFilter: ["data-theme", "data-palette"] });
  addEventListener("scroll", scroll, { passive: true });
  if (canvas && !reduced) { hero.addEventListener("pointermove", move); hero.addEventListener("pointerleave", leave); }
  addEventListener("resize", resize);
  document.fonts.ready.then(resize);
  resize(); onScroll();
})();

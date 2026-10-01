(() => {
  "use strict";

  const masthead = document.querySelector("[data-masthead]");
  if (!masthead) return;
  const bar = masthead.querySelector(".masthead__bar");

  const TRAVEL = 48;
  let lastY = window.scrollY;
  let travel = 0;

  function update() {
    const y = window.scrollY;
    const step = y - lastY;
    lastY = y;

    const docked = masthead.getBoundingClientRect().bottom <= bar.offsetHeight + 1;
    masthead.classList.toggle("is-docked", docked);

    if (Math.abs(step) >= window.innerHeight) travel = 0;
    else if (step !== 0) travel = Math.sign(step) === Math.sign(travel) ? travel + step : step;

    const vmin = Math.min(window.innerWidth, window.innerHeight) / 100;
    if (!docked || y < masthead.offsetHeight + 15 * vmin) {
      masthead.classList.remove("is-hidden");
    } else if (Math.abs(travel) >= TRAVEL) {
      masthead.classList.toggle("is-hidden", travel > 0);
    }
  }

  window.addEventListener("scroll", () => requestAnimationFrame(update), { passive: true });
  window.addEventListener("resize", update);
  update();

  window.addEventListener("load", () =>
    requestAnimationFrame(() => {
      update();
      masthead.classList.add("is-ready");
    })
  );
})();

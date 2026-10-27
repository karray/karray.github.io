// loaded without defer at the top of <head>, so the theme is set before the first paint
(() => {
  const root = document.documentElement, system = matchMedia("(prefers-color-scheme: dark)");
  const saved = () => { try { return localStorage.getItem("theme"); } catch { return null; } };
  function apply() {
    const t = saved();
    root.dataset.mode = t === "dark-mode" ? "dark" : t === "light-mode" ? "light" : "auto";
    root.dataset.theme = root.dataset.mode === "auto" ? (system.matches ? "dark" : "light") : root.dataset.mode;
    document.querySelectorAll(".theme-toggle").forEach((b) => b.setAttribute("aria-label", `Theme: ${root.dataset.mode}`));
  }
  system.addEventListener("change", apply);
  addEventListener("storage", apply);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".theme-toggle")) return;
    const next = { auto: "light-mode", light: "dark-mode" }[root.dataset.mode];
    try { next ? localStorage.setItem("theme", next) : localStorage.removeItem("theme"); } catch {}
    apply();
  });
  apply();
  document.addEventListener("DOMContentLoaded", apply);
})();

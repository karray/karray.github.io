// The light or dark theme shared by the demos. A visitor's choice is remembered for all demos;
// without one, the system setting is used. Load this at the start of <body>, so that the page is
// drawn in the right theme from the start; a button #theme-toggle-button switches the theme.
(() => {
  let saved = null;
  try {
    saved = localStorage.getItem("theme");
  } catch {}
  const dark = saved ? saved === "dark-mode" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.body.classList.toggle("dark-mode", dark);

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle-button");
    if (!button) return;
    button.setAttribute("aria-pressed", dark);
    button.addEventListener("click", () => {
      const isDark = document.body.classList.toggle("dark-mode");
      button.setAttribute("aria-pressed", isDark);
      try {
        localStorage.setItem("theme", isDark ? "dark-mode" : "light-mode");
      } catch {}
    });
  });
})();

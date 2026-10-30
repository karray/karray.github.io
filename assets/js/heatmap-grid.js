(() => {
  "use strict";

  if (window.HeatmapGrid) return;

  class HeatmapGrid {
    constructor(root) {
      this.root = root;
      this.bindOpacity();
      this.tiles = new window.TileReveal(root, {
        prefix: "hg",
        autoArm: true,
        getCols: () => {
          const grid = root.querySelector(".hg-grid");
          return grid
            ? Math.max(1, getComputedStyle(grid).gridTemplateColumns.split(" ").length)
            : 1;
        },
      });
      this.maps = this.tiles.maps;
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

/* Display precomputed real-image results. No inference runs in the reader's browser. */
(() => {
  "use strict";
  const gallery = document.getElementById("lf-gallery");
  if (!gallery) return;
  const data = JSON.parse(
    document.getElementById("lf-gallery-data").textContent,
  );
  const imageBase = gallery.dataset.images;
  const heatmapBase = gallery.dataset.heatmaps;
  const panels = document.getElementById("lf-panels");
  const opacity = document.getElementById("lf-opacity");
  const reveal = document.getElementById("lf-reveal");
  let imageIndex = 0;

  const imagePath = (image) => `${imageBase}/${image.id}.${image.extension}`;
  const heatmapPath = (image, model, method) =>
    `${heatmapBase}/${method}_${model.id}_${image.id}.png`;

  const bindReveal = (stack) => {
    stack.addEventListener("pointerenter", () => {
      if (reveal.checked) stack.classList.add("lf-hover-hide");
    });
    stack.addEventListener("pointerleave", () =>
      stack.classList.remove("lf-hover-hide"),
    );
    stack.addEventListener("pointermove", (event) => {
      const bounds = stack.getBoundingClientRect();
      stack.style.setProperty("--lf-x", `${event.clientX - bounds.left}px`);
      stack.style.setProperty("--lf-y", `${event.clientY - bounds.top}px`);
    });
  };

  const render = () => {
    const image = data.images[imageIndex];
    const mapOpacity = Number(opacity.value) / 100;
    document.getElementById("lf-image-title").textContent = image.title;
    document.getElementById("lf-image-count").textContent =
      `${imageIndex + 1} / ${data.images.length}`;
    document.getElementById("lf-opacity-value").textContent =
      `${Math.round(mapOpacity * 100)}%`;
    const header = document.createElement("div");
    header.className = "lf-grid-header";
    header.innerHTML =
      "<span>Model</span><span>Original</span><span>LaFAM</span><span>AnyUp</span>";
    panels.replaceChildren(
      header,
      ...data.models.map((model) => {
        const panel = document.createElement("div");
        panel.className = "lf-grid-row";
        panel.innerHTML = `<strong class="lf-panel-title">${model.label}</strong><div class="lf-grid-cell"><div class="lf-stack"><img class="lf-input" src="${imagePath(image)}" alt="${image.title}"></div></div>${["lafam", "anyup"].map((method) => `<div class="lf-grid-cell"><div class="lf-stack"><img class="lf-input" src="${imagePath(image)}" alt="${image.title}"><img class="lf-map" src="${heatmapPath(image, model, method)}" alt="${method} heatmap from ${model.label}"></div></div>`).join("")}`;
        panel.querySelectorAll(".lf-stack").forEach((stack) => {
          const map = stack.querySelector(".lf-map");
          if (map) map.style.opacity = mapOpacity;
          bindReveal(stack);
        });
        return panel;
      }),
    );
    document.getElementById("lf-gallery-status").textContent =
      `${image.title} · LaFAM and AnyUp across ten encoders. Move over a view to reveal the image beneath the map.`;
  };

  document.getElementById("lf-previous").addEventListener("click", () => {
    imageIndex = (imageIndex + data.images.length - 1) % data.images.length;
    render();
  });
  document.getElementById("lf-next").addEventListener("click", () => {
    imageIndex = (imageIndex + 1) % data.images.length;
    render();
  });
  opacity.addEventListener("input", render);
  reveal.addEventListener("change", render);
  render();
})();

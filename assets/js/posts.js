function getScroll() {
  return getMinViewHeightWidth(
    window.pageYOffset || document.documentElement.scrollTop
  );
}

function getMinViewHeightWidth(x) {
  return (x / Math.min(window.innerHeight, window.innerWidth)) * 100;
}

let header = document.getElementById("post-header");
const unit = "vmin";

const header_height = 60;

let is_small_screen = window.matchMedia("(max-width: 350px)");
let prevScroll = 0;

function update_elements() {
  // if(is_small_screen.matches){
  //     header.classList.add('fixed-header')
  //     return
  // }

  let scroll = getScroll();
  let new_height = header_height - scroll;
  header.style.height = new_height + unit;

  if (new_height < 15) {
    header.classList.add("fixed-header");
    // hide the header if scrolling down
    if (new_height < -15 && scroll > prevScroll) {
      header.classList.add("hidden-header");
    } else {
      header.classList.remove("hidden-header");
    }
    prevScroll = scroll;
  } else {
    header.classList.remove("fixed-header");
  }
}

window.addEventListener("scroll", () => requestAnimationFrame(update_elements));
window.addEventListener(
  "resize",
  function (event) {
    update_elements();
  },
  true
);

update_elements();

(() => {
  console.log("Inint popup...");
  const refRoot = document.getElementById("reference-list");
  if (!refRoot) return;
  console.log("Found references...", refRoot);
  // Build reference index from the list
  const refIndex = new Map();
refRoot.querySelectorAll('.ref').forEach((div) => {
    console.log("Indexing ref...", div);
    if (!div.id) return;
    refIndex.set(div.id, {
        id: div.id,
        index: Number(div.dataset.index) || "?",
        author: div.dataset.author || "",
        year: div.dataset.year || "",
        title: div.dataset.title || "",
        node: div,
    });
});
  console.log("Built index...", refIndex);

  // Helpers
  const clean = (s) => (s || "").toLowerCase().replace(/[^a-z]+/g, ""); // letters only
  const keyOf = (t) => clean(t);

  const cache = new Map(); // cleanedTitle -> Promise<metadata|null>

  // Data providers (sequential, minimal pattern)
  const providers = [
    async function crossrefProvider(key, rawTitle) {
      const url =
        "https://api.crossref.org/works?query.title=" +
        encodeURIComponent(rawTitle) +
        "&rows=5";
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error("crossref http");
      const j = await r.json();
      const items = (j.message && j.message.items) || [];
      let best =
        items.find((it) => keyOf((it.title && it.title[0]) || "") === key) ||
        items[0];
      if (!best) return null;
      const abstract = best.abstract
        ? best.abstract.replace(/<[^>]+>/g, "").trim()
        : null;
      const urlBest =
        best.URL || (best.link && best.link[0] && best.link[0].URL) || null;
      return {
        abstract,
        citations:
          typeof best["is-referenced-by-count"] === "number"
            ? best["is-referenced-by-count"]
            : null,
        url: urlBest,
        source: "Crossref",
      };
    },
    async function openalexProvider(key, rawTitle) {
      const url =
        "https://api.openalex.org/works?search=" +
        encodeURIComponent(rawTitle) +
        "&per_page=5";
      const r = await fetch(url);
      if (!r.ok) throw new Error("openalex http");
      const j = await r.json();
      const items = j.results || [];
      let best =
        items.find((it) => keyOf(it.display_name || "") === key) || items[0];
      if (!best) return null;
      const iii = best.abstract_inverted_index;
      const abstract = iii
        ? (() => {
            const words = Object.keys(iii);
            const max = Math.max(...words.flatMap((w) => iii[w]));
            const arr = new Array(max + 1);
            for (const w of words) for (const pos of iii[w]) arr[pos] = w;
            return arr.join(" ");
          })()
        : null;
      const urlBest =
        best.primary_location?.source?.host_venue_url ||
        best.primary_location?.landing_page_url ||
        best.open_access?.oa_url ||
        best.doi ||
        best.id;
      return {
        abstract,
        citations:
          typeof best.cited_by_count === "number" ? best.cited_by_count : null,
        url: urlBest,
        source: "OpenAlex",
      };
    },
  ];

  async function getMetadata(title) {
    const key = keyOf(title);
    if (cache.has(key)) return cache.get(key);
    const p = (async () => {
      for (const provider of providers) {
        try {
          const res = await provider(key, title);
          if (res && (res.abstract || res.citations != null || res.url))
            return res;
        } catch (e) {
          /* try next */
        }
      }
      return null;
    })();
    cache.set(key, p);
    return p;
  }

  // Popup
  const popup = document.createElement("div");
  popup.className = "cite-popup";
  popup.hidden = true;
  popup.setAttribute("role", "dialog");
  document.body.appendChild(popup);

  function positionPopup(target) {
    const r = target.getBoundingClientRect();
    popup.style.left = r.left + window.scrollX + "px";
    popup.style.top = r.bottom + window.scrollY + 6 + "px";
  }

  function showPopup(target, ref) {
    const { author, year, title, index } = ref;
    popup.innerHTML = `
    <h4>${title}</h4>
    <div class="meta">${author} ${year ? "(" + year + ")" : ""}</div>
    <div class="abstract loading">Loading abstract...</div>
    <div class="links"></div>
    `;
    positionPopup(target);
    popup.hidden = false;
    getMetadata(title).then((meta) => {
      const absEl = popup.querySelector(".abstract");
      const linksEl = popup.querySelector(".links");
      if (!meta) {
        absEl.textContent = "";
        absEl.classList.remove("loading");
        return;
      }
      absEl.textContent = meta.abstract || "No abstract available.";
      absEl.classList.remove("loading");
      const link = meta.url
        ? `<a href="${meta.url}" target="_blank" rel="noopener">Open paper</a>`
        : "";
      const src = meta.source
        ? `<span style="margin-left:8px;font-size:12px;opacity:.8">from ${
            meta.source
          }${
            meta.citations != null ? " • citations: " + meta.citations : ""
          }</span>`
        : "";
      linksEl.innerHTML = link + src;
    });
  }

  function hidePopup() {
    popup.hidden = true;
  }

  function bindAnchor(a) {
    console.log("Binding anchor...", a);
    const rid = a.getAttribute("data-ref");
    const ref = refIndex.get(rid);
    console.log(ref);
    if (!ref) return;
    console.log("... found reference", ref);
    a.textContent = `[${ref.index}]`;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      showPopup(a, ref);
    });
  }

  document.querySelectorAll("a[data-ref]").forEach(bindAnchor);

  // Make the reference list itself clickable
  for (const ref of refIndex.values()) {
    ref.node.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      showPopup(ref.node, ref);
    });
  }

  // Dismissal
  document.addEventListener("click", (e) => {
    if (!popup.hidden && !popup.contains(e.target)) hidePopup();
  });
  window.addEventListener("resize", () => {
    if (!popup.hidden) hidePopup();
  });
})();

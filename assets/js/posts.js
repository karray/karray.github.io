(() => {
  const refRoot = document.getElementById("reference-list");
  if (!refRoot) return;
  // Build reference index from the list
  const refIndex = new Map();
  refRoot.querySelectorAll(".ref").forEach((div) => {
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

  // Helpers
  // letters only, with accents folded so "déjà" matches "deja"
  const clean = (s) =>
    (s || "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^a-z]+/g, "");
  const keyOf = (t) => clean(t);

  // Dice coefficient over letter bigrams: tolerates small differences such as
  // "Map" vs "Maps", but rejects a different paper that shares a few words.
  const bigrams = (s) => {
    const m = new Map();
    for (let i = 0; i < s.length - 1; i++) {
      const b = s.slice(i, i + 2);
      m.set(b, (m.get(b) || 0) + 1);
    }
    return m;
  };
  function similarity(a, b) {
    if (a === b) return 1;
    const A = bigrams(a),
      B = bigrams(b);
    let shared = 0;
    for (const [k, c] of A) shared += Math.min(c, B.get(k) || 0);
    return (2 * shared) / (a.length + b.length - 2 || 1);
  }
  const MIN_SIMILARITY = 0.9;

  // Plain text without markup or a leading "Abstract" heading, or null
  const tidyAbstract = (s) =>
    (s || "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/^abstract\b[\s.:]*/i, "") || null;

  // The candidate whose title is closest to the reference's, or null when none
  // is close enough; a search engine's top hit is not necessarily the paper.
  // Titles alone can mislead ("More Sanity Checks for Saliency Maps" is 0.93
  // like "Sanity Checks for Saliency Maps"), so a candidate that lists its
  // authors must also name the reference's first author among them.
  function bestMatch(ref, items, titleOf, authorsOf) {
    const surname = clean(ref.author);
    let best = null,
      bestScore = MIN_SIMILARITY;
    for (const it of items) {
      const score = similarity(ref.key, keyOf(titleOf(it)));
      if (score < bestScore) continue;
      const authors = authorsOf(it).map(clean);
      if (surname && authors.length && !authors.some((a) => a.includes(surname)))
        continue;
      (best = it), (bestScore = score);
    }
    return best;
  }

  const cache = new Map(); // cleanedTitle -> lookup, see lookup()

  async function crossrefProvider(ref) {
    const url =
      "https://api.crossref.org/works?query.title=" +
      encodeURIComponent(ref.title) +
      "&rows=5";
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error("crossref http");
    const j = await r.json();
    const items = (j.message && j.message.items) || [];
    const best = bestMatch(
      ref,
      items,
      (it) => (it.title && it.title[0]) || "",
      (it) => (it.author || []).map((a) => a.family || a.name || "")
    );
    if (!best) return null;
    const abstract = tidyAbstract(best.abstract);
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
  }

  // arXiv's own API sends no CORS headers, but every arXiv paper has a DataCite
  // DOI whose record carries the abstract. Citation counts there are not
  // tracked, so it gives none.
  async function arxivProvider(ref) {
    // lower case, so that "and", "or" and "not" are not read as operators,
    // and no punctuation, which the query syntax reserves
    const query = ref.title.toLowerCase().replace(/[^\p{L}\p{N}\s]+/gu, " ");
    const url =
      "https://api.datacite.org/dois?client-id=arxiv.content&page%5Bsize%5D=5" +
      "&fields%5Bdois%5D=doi,url,titles,creators,descriptions&query=" +
      encodeURIComponent("titles.title:(" + query + ")");
    const r = await fetch(url);
    if (!r.ok) throw new Error("datacite http");
    const j = await r.json();
    const items = (j.data || []).map((d) => d.attributes || {});
    const best = bestMatch(
      ref,
      items,
      (it) => it.titles?.[0]?.title || "",
      (it) => (it.creators || []).map((c) => c.familyName || c.name || "")
    );
    if (!best) return null;
    const abstract = (best.descriptions || []).find(
      (d) => d.descriptionType === "Abstract"
    );
    return {
      abstract: tidyAbstract(abstract && abstract.description),
      citations: null,
      url: best.url || (best.doi && "https://doi.org/" + best.doi) || null,
      source: "arXiv",
    };
  }

  async function openalexProvider(ref) {
    // "?" and "*" are wildcards in OpenAlex search syntax
    const query = ref.title.replace(/[^\p{L}\p{N}\s-]+/gu, " ");
    const url =
      "https://api.openalex.org/works?search=" +
      encodeURIComponent(query) +
      "&per_page=5";
    const r = await fetch(url);
    if (!r.ok) throw new Error("openalex http");
    const j = await r.json();
    const items = j.results || [];
    const best = bestMatch(
      ref,
      items,
      (it) => it.display_name || "",
      (it) => (it.authorships || []).map((a) => a.author?.display_name || "")
    );
    if (!best) return null;
    const iii = best.abstract_inverted_index;
    const abstract = iii
      ? (() => {
          const words = Object.keys(iii);
          const max = Math.max(...words.flatMap((w) => iii[w]));
          const arr = new Array(max + 1);
          for (const w of words) for (const pos of iii[w]) arr[pos] = w;
          return tidyAbstract(arr.join(" "));
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
  }

  // Has abstracts that the others lack (e.g. Springer, Elsevier).
  async function semanticScholarProvider(ref) {
    const url =
      "https://api.semanticscholar.org/graph/v1/paper/search/match?query=" +
      encodeURIComponent(ref.title) +
      "&fields=title,authors,abstract,citationCount,url,externalIds";
    const r = await fetch(url);
    if (r.status === 404) return null; // no paper with a matching title
    if (!r.ok) throw new Error("semantic scholar http");
    const j = await r.json();
    const best = bestMatch(
      ref,
      j.data || [],
      (it) => it.title || "",
      (it) => (it.authors || []).map((a) => a.name || "")
    );
    if (!best) return null;
    const ids = best.externalIds || {};
    const urlBest =
      (ids.DOI && "https://doi.org/" + ids.DOI) ||
      (ids.ArXiv && "https://arxiv.org/abs/" + ids.ArXiv) ||
      best.url ||
      null;
    return {
      abstract: tidyAbstract(best.abstract),
      citations:
        typeof best.citationCount === "number" ? best.citationCount : null,
      url: urlBest,
      source: "Semantic Scholar",
    };
  }

  // Sources asked in turn, a tier at a time, until every field is filled; the
  // sources within a tier are asked together. Where two have the same field,
  // the earlier one wins, so the published version beats the preprint.
  //  1. Crossref and arXiv have no tight limits and between them usually cover
  //     everything: the publisher's link and citations, and the abstract.
  //  2. OpenAlex limits keyless search to a daily quota for each visitor.
  //  3. Semantic Scholar's keyless limit is shared by everyone, so it is only
  //     asked for what the others did not have.
  const tiers = [
    [crossrefProvider, arxivProvider],
    [openalexProvider],
    [semanticScholarProvider],
  ];

  const FIELDS = ["abstract", "citations", "url"];

  // Starts asking the sources about a reference and returns at once, so the
  // popup can show each field as it arrives instead of waiting for the slowest
  // source. A field keeps the value of the earliest source in the tiers that
  // has it, so a later source cannot hide an abstract found elsewhere, and a
  // quicker one only holds its place until a preferred one answers.
  // Listeners hear of every change, the last time with done set.
  function lookup({ title, author }) {
    const key = keyOf(title);
    if (cache.has(key)) return cache.get(key);
    const ref = { key, title, author };
    const found = {}; // field -> { value, rank, source }
    const entry = {
      meta: { abstract: null, citations: null, url: null, source: "" },
      done: false,
      listeners: new Set(),
    };
    const publish = () => {
      const meta = {};
      for (const f of FIELDS) meta[f] = found[f] ? found[f].value : null;
      const byRank = Object.values(found).sort((a, b) => a.rank - b.rank);
      meta.source = [...new Set(byRank.map((x) => x.source))].join(" + ");
      entry.meta = meta;
      entry.listeners.forEach((fn) => fn(entry));
    };
    const merge = (res, rank) => {
      let changed = false;
      for (const f of FIELDS) {
        if (res[f] == null || (found[f] && found[f].rank < rank)) continue;
        found[f] = { value: res[f], rank, source: res.source };
        changed = true;
      }
      if (changed) publish();
    };
    (async () => {
      let failed = false,
        rank = 0;
      for (const tier of tiers) {
        await Promise.all(
          tier.map((provider) => {
            const r = rank++;
            return provider(ref).then(
              (res) => res && merge(res, r),
              () => (failed = true)
            );
          })
        );
        if (FIELDS.every((f) => found[f])) break;
      }
      // a source that failed (network error, rate limit) may still have the
      // abstract, so ask again the next time the popup opens
      if (failed && !found.abstract) cache.delete(key);
      entry.done = true;
      publish();
    })();
    cache.set(key, entry);
    return entry;
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

  let shown = null; // the lookup whose reference the popup is showing

  function render(entry) {
    if (entry !== shown) return; // the popup has moved on to another reference
    const { meta, done } = entry;
    const absEl = popup.querySelector(".abstract");
    const linksEl = popup.querySelector(".links");
    absEl.classList.toggle("loading", !meta.abstract && !done);
    absEl.textContent =
      meta.abstract ||
      (!done ? "Loading abstract..." : meta.source ? "No abstract available." : "");
    const link = meta.url
      ? `<a href="${meta.url}" target="_blank" rel="noopener">Open paper</a>`
      : "";
    const src = meta.source
      ? `<span style="margin-left:8px;font-size:12px;opacity:.8">from ${
          meta.source
        }${meta.citations != null ? " • citations: " + meta.citations : ""}</span>`
      : "";
    linksEl.innerHTML = link + src;
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
    if (shown) shown.listeners.delete(render);
    shown = lookup(ref);
    shown.listeners.add(render);
    render(shown);
  }

  function hidePopup() {
    popup.hidden = true;
    if (shown) shown.listeners.delete(render);
    shown = null;
  }

  function bindAnchor(a) {
    console.log("Binding anchor...", a);
    const rid = a.getAttribute("data-ref");
    const ref = refIndex.get(rid);
    console.log(ref);
    if (!ref) return;
    console.log("... found reference", ref);
    a.textContent = `${ref.index}`;
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

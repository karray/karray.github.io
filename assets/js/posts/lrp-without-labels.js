(async function () {
  'use strict';
  const root = document.getElementById('lrp-article');
  if (!root) return;
  const widgets = [...document.querySelectorAll('.lrp-widget')];
  try {
    const response = await fetch(root.dataset.source);
    if (!response.ok) throw new Error('Data request failed');
    const data = await response.json();
    const control = (w, name) => w.querySelector(`[data-control="${name}"]`);
    const role = (w, name) => w.querySelector(`[data-role="${name}"]`);
    const text = (w, name, value) => { role(w, name).textContent = value; };
    const number = x => (x >= 0 ? '+' : '−') + Math.abs(x).toPrecision(5);
    const options = (select, entries) => entries.forEach(([value, label]) => {
      const o = document.createElement('option'); o.value = value; o.textContent = label; select.append(o);
    });
    const scale = (w, h, positive = false) => text(w, 'scale', `${h.shape.join(' × ')} · ${positive ? 'black 0 → yellow/white +' : 'blue − / white 0 / red +'}${h.peak.toExponential(2)} (independent scale)`);
    const selected = w => w.querySelector('.fg-thumb[aria-selected="true"]');
    const imageName = id => Object.keys(data.images).find(k => k.split('.')[0] === id);

    const g = document.getElementById('lrp-gallery');
    const gg = document.getElementById('lrp-model-maps');
    options(control(g, 'model'), Object.entries(data.models).map(([k, v]) => [k, v.label]));
    control(g, 'model').value = 'dino_vit';
    function gallery() {
      const name = imageName(selected(gg).dataset.item);
      const m = data.models[control(g, 'model').value], x = m.images[name], a = x.accounting;
      text(g, 'scale', `Independent maxima: signed ±${x.signed.peak.toExponential(2)}; positive-only ${x.positive.peak.toExponential(2)}. Displayed image: ${data.images[name].label}.`);
      text(g, 'ledger', `Initial ${number(a.initial.net)} → pixels ${number(a.pixels.net)} (positive ${number(a.pixels.positive)}, negative ${number(a.pixels.negative)}); prefix tokens ${number(a.prefix_tokens.net)}; signed rule gaps ${number(a.rule_gap_sum)}. Residual rule: ${m.spec.residual}; projection: ${m.spec.stem}.`);
    }
    // The gallery's own generic axis control already moves the Model dropdown's
    // dataset onto every panel and crossfades; this listener only refreshes text.
    control(g, 'model').addEventListener('change', gallery);
    new MutationObserver(gallery).observe(gg.querySelector('.fg-dock'), {subtree:true,attributes:true,attributeFilter:['aria-selected']});
    gallery();

    const s = document.getElementById('lrp-stem');
    const sg = document.getElementById('lrp-projection-maps');
    function stem() {
      const model = selected(sg).dataset.item, x = data.stems[model];
      text(s, 'scale', `Independent maxima: WSquare ±${x.wsquare.peak.toExponential(2)}; ZPlus ±${x.zplus.peak.toExponential(2)}.`);
      text(s, 'result', `One-template energy: WSquare ${(100*x.rank1.wsquare).toFixed(1)}% → ZPlus ${(100*x.rank1.zplus).toFixed(1)}%. Signed pixel totals: ${number(x.wsquare.net)} → ${number(x.zplus.net)}. Identical incoming patch relevance.`);
    }
    new MutationObserver(stem).observe(sg.querySelector('.fg-dock'), {subtree:true,attributes:true,attributeFilter:['aria-selected']});
    stem();

    const t = document.getElementById('lrp-trace');
    const tg = document.getElementById('lrp-backward-maps');
    function trace() {
      const name = imageName(selected(tg).dataset.item), allocation = control(t, 'allocation').value;
      const x = data.mae[name][allocation], index = Number(control(t, 'stage').value), h = x.stages[index], n = x.norm;
      scale(t, h);
      text(t, 'stage', `${index + 1} / ${x.stages.length} · ${h.label}`);
      control(t, 'stage').setAttribute('aria-valuetext', h.label);
      text(t, 'amount', `Initial total ${number(n.initial_net)} · displayed spatial net ${number(h.net)} · incoming-patch / summed-pixel correlation ${x.patch_correlation.toFixed(5)}. Stage totals exclude CLS and other destinations.`);
      text(t, 'bias', `Exact output decomposition for this allocation: image-dependent term ${number(n.image_term_net)} + learned-bias contribution ${number(n.bias_term_net)} = initial total ${number(n.initial_net)}. This is an output decomposition, not a claim of exact backward pixel conservation.`);
    }
    control(t, 'allocation').addEventListener('change', trace);
    control(t, 'stage').addEventListener('input', trace);
    new MutationObserver(trace).observe(tg.querySelector('.fg-dock'), {subtree:true,attributes:true,attributeFilter:['aria-selected']});
    trace();

    const i = document.getElementById('lrp-intermediate');
    const ig = document.getElementById('lrp-intermediate-maps');
    function intermediate() {
      const item = imageName(selected(ig).dataset.item);
      const stages = data.intermediate.models[control(i, 'model').value].images[item.split('.')[0]].stages;
      const index = Number(control(i, 'depth').value), h = stages[index];
      scale(i, h, true);
      text(i, 'stage', `Depth ${index} / 4 · ${h.label}`);
      control(i, 'depth').setAttribute('aria-valuetext', h.label);
      text(i, 'amount', `Phase ratio ${h.phase_ratio.toFixed(2)} · displayed spatial net ${number(h.net)} · ${(100 * h.nonpositive_after_mean).toFixed(1)}% of positions nonpositive after the channel mean (rendered black here, not necessarily zero).`);
    }
    control(i, 'model').addEventListener('change', intermediate);
    control(i, 'depth').addEventListener('input', intermediate);
    new MutationObserver(intermediate).observe(ig.querySelector('.fg-dock'), {subtree:true,attributes:true,attributeFilter:['aria-selected']});
    intermediate();

    const c = document.getElementById('lrp-checkerboard');
    const cg = document.getElementById('lrp-checkerboard-maps');
    function checkerboard() {
      const itemId = selected(cg).dataset.item, label = data.images[imageName(itemId)].label;
      const m = data.checkerboard.models[control(c, 'model').value].images[itemId];
      const share = x => x.main.net / x.sum.net;
      text(c, 'scale', `Independent maxima per panel. Main and shortcut sum exactly to the block input (additivity error ${m.reference.additivity_relative_error.toExponential(1)} unsplit, ${m.magnitude.additivity_relative_error.toExponential(1)} split by magnitude).`);
      text(c, 'result', `On ${label}, main-branch share of block-input relevance mass: ${(100 * share(m.reference)).toFixed(1)}% unsplit → ${(100 * share(m.magnitude)).toFixed(1)}% split by magnitude. This post's own default residual rule for ResNet, equal, is scale-equivalent to unsplit—it would show the same share as the "Unsplit" columns.`);
    }
    control(c, 'model').addEventListener('change', checkerboard);
    new MutationObserver(checkerboard).observe(cg.querySelector('.fg-dock'), {subtree:true,attributes:true,attributeFilter:['aria-selected']});
    checkerboard();

    widgets.forEach(w => { w.dataset.ready = 'true'; });
  } catch (error) {
    widgets.forEach(w => { const p = document.createElement('p'); p.className = 'lrp-error'; p.textContent = 'Interactive data could not load. The article and downloadable reports remain available; reload to retry.'; w.append(p); });
    console.error(error);
  }
})();

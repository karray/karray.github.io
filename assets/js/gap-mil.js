/* Educational arithmetic only: no model inference, dependencies, or network calls. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const winner = (dog, bicycle) => dog === bicycle ? 'tie' : dog > bicycle ? 'dog' : 'bicycle';
  const count = byId('mil-count');
  if (count) {
    // Fill from the center out to show a contiguous growing region.
    const order = Array.from({length:36}, (_, i) => i).sort((a, b) => {
      const distance = i => (Math.floor(i / 6) - 2.5) ** 2 + (i % 6 - 2.5) ** 2;
      return distance(a) - distance(b) || a - b;
    });
    const update = () => {
      const n = Number(count.value);
      const selected = new Set(order.slice(0, n));
      const grid = byId('mil-grid');
      const cells = Array.from({length:36}, (_, i) => {
        const cell = document.createElement('span');
        cell.textContent = selected.has(i) ? 'D' : 'B';
        if (!selected.has(i)) cell.className = 'mil-bike-cell';
        cell.setAttribute('aria-hidden', 'true');
        return cell;
      });
      grid.replaceChildren(...cells);
      grid.setAttribute('aria-label', `36 positions: ${n} support dog and ${36-n} support bicycle`);
      byId('mil-count-value').textContent = `${n} of 36`;
      const dog = n * 6 / 36;
      const bicycle = (36 - n) * 2 / 36;
      byId('mil-dog-value').textContent = dog.toFixed(2);
      byId('mil-bike-value').textContent = bicycle.toFixed(2);
      byId('mil-dog-bar').style.width = `${dog / 6 * 100}%`;
      byId('mil-bike-bar').style.width = `${bicycle / 6 * 100}%`;
      const result = winner(dog, bicycle);
      byId('mil-area-result').textContent = `${result === 'tie' ? 'The GAP scores tie.' : `GAP predicts ${result}.`} All ${n} dog positions still predict dog locally.`;
    };
    count.disabled = false;
    count.addEventListener('input', update);
    update();
  }
  const spike = byId('mil-spike');
  if (spike) {
    const update = () => {
      const s = Number(spike.value);
      byId('mil-spike-value').textContent = String(s);
      byId('mil-mean-result').textContent = `Dog: ${(16 / 9).toFixed(2)} · Bicycle: ${(s / 9).toFixed(2)} → ${winner(16, s)}`;
      byId('mil-max-result').textContent = `Dog: 2.00 · Bicycle: ${s.toFixed(2)} → ${winner(2, s)}`;
    };
    spike.disabled = false;
    spike.addEventListener('input', update);
    update();
  }
})();

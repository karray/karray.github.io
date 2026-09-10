/* Original educational geometry; no inference, analytics, or network requests. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const fixed = n => {
    const rounded = Math.round((Math.abs(n) + Number.EPSILON) * 100) / 100;
    return (rounded === 0 ? 0 : n < 0 ? -rounded : rounded).toFixed(2);
  };
  const winner = (a, b) => Math.abs(a - b) < 1e-10 ? 'tie' : a > b ? 'dog' : 'competitor';
  const setAttrs = (node, attrs) => Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  const length = el('cv-length');
  const angle = el('cv-angle');
  if (length && angle) {
    const update = () => {
      const r = Number(length.value);
      const degrees = Number(angle.value);
      const theta = degrees * Math.PI / 180;
      const dog = r * Math.cos(theta);
      const cat = r * Math.sin(theta);
      const x = 180 + 50 * dog;
      const y = 225 - 50 * cat;
      el('cv-length-value').textContent = r.toFixed(1);
      el('cv-angle-value').textContent = r === 0 ? 'undefined (zero vector)' : `${degrees}°`;
      el('cv-dog-score').textContent = fixed(dog);
      el('cv-cat-score').textContent = fixed(cat);
      el('cv-alignment').textContent = r === 0 ? 'undefined' : fixed(Math.cos(theta));
      setAttrs(el('cv-feature-arrow'), {x2:x, y2:y, visibility:r === 0 ? 'hidden' : 'visible'});
      setAttrs(el('cv-feature-tip'), {cx:x, cy:y});
      setAttrs(el('cv-projection-drop'), {x1:x, y1:y, x2:x});
      setAttrs(el('cv-projection'), {x2:x});
      const result = winner(dog, cat);
      el('cv-vector-result').textContent = r === 0 ? 'Both scores are zero. A zero vector has no direction.' : `${result === 'tie' ? 'The scores tie.' : result === 'dog' ? 'Dog wins.' : 'Cat wins.'} Stretching this arrow scales both scores; turning it changes their balance.`;
      el('cv-vector-desc').textContent = `Length ${r}, ${r === 0 ? 'direction undefined' : `angle ${degrees} degrees`}. Dog score ${fixed(dog)}, cat score ${fixed(cat)}.`;
    };
    [length, angle].forEach(input => { input.disabled = false; input.addEventListener('input', update); });
    update();
  }

  const strength = el('cv-strength');
  if (strength) {
    let mode = 'quiet';
    const objectCells = new Set([5,6,9,10]);
    const color = (value, max, purple) => {
      const t = Math.max(0, Math.min(1, value / max));
      const start = [238,243,237];
      const end = purple ? [113,81,142] : [20,104,95];
      return `rgb(${start.map((v,i) => Math.round(v + (end[i] - v) * t)).join(',')})`;
    };
    const drawMap = (id, fg, bg, max, purple, label) => {
      const cells = Array.from({length:16}, (_,i) => {
        const cell = document.createElement('span');
        const value = objectCells.has(i) ? fg : bg;
        if (objectCells.has(i)) cell.className = 'cv-object';
        cell.textContent = fixed(value);
        cell.style.backgroundColor = color(value, max, purple);
        cell.style.color = value / max >= 0.63 ? '#ffffff' : '#22372c';
        cell.setAttribute('aria-hidden', 'true');
        return cell;
      });
      el(id).replaceChildren(...cells);
      el(id).setAttribute('aria-label', `${label}: four object positions ${fixed(fg)}; twelve background positions ${fixed(bg)}.`);
    };
    const update = () => {
      const t = Number(strength.value) / 100;
      const a = 0.8 * t;
      const x = mode === 'quiet' ? 0 : 3 * a;
      const y = mode === 'quiet' ? 1.5 * (1 - 0.85 * t) : 1.5 * (1 - a);
      const norm = Math.hypot(x, y);
      const cosine = x / norm;
      const dog = (4 * 3 + 12 * x) / 16;
      const other = 12 * y / 16;
      el('cv-strength-value').textContent = `${strength.value}%`;
      drawMap('cv-map-norm',3,norm,3,false,'Feature lengths');
      drawMap('cv-map-angle',1,cosine,1,true,'Cosine with dog direction');
      drawMap('cv-map-score',3,x,3,false,'Dog scores');
      el('cv-field-detail').textContent = mode === 'quiet' ? `Background length ${t === 0 ? 'starts at' : 'falls to'} ${fixed(norm)}, but dog alignment stays 0.00. Its dog score remains 0.00.` : `Background length is ${fixed(norm)} and dog alignment is ${fixed(cosine)}. Together they give dog score ${fixed(x)} at each background position.`;
      el('cv-field-dog').textContent = fixed(dog);
      el('cv-field-other').textContent = fixed(other);
      el('cv-field-winner').textContent = winner(dog,other);
    };
    for (const name of ['quiet','context']) {
      const button = el(`cv-mode-${name}`);
      button.disabled = false;
      button.addEventListener('click', () => {
        mode = name;
        for (const other of ['quiet','context']) el(`cv-mode-${other}`).setAttribute('aria-pressed',String(other === mode));
        update();
      });
    }
    strength.disabled = false;
    strength.addEventListener('input', update);
    update();
  }

  const gain = el('cv-gain');
  if (gain) {
    const update = () => {
      const g = Number(gain.value);
      const a = Math.sqrt(1.5 * g * g + 1.5);
      const b = Math.sqrt(0.5 * g * g + 2.5);
      el('cv-gain-value').textContent = `${g.toFixed(1)}×`;
      el('cv-norm-a').textContent = fixed(a);
      el('cv-norm-b').textContent = fixed(b);
      el('cv-norm-a-bar').style.width = `${a / 5.3 * 100}%`;
      el('cv-norm-b-bar').style.width = `${b / 5.3 * 100}%`;
      el('cv-ln-result').textContent = g === 1 ? 'With every gain set to 1, both vectors keep their standardized length of √3.' : 'The same gain produces different lengths because the vectors put different amounts into the first channel.';
    };
    gain.disabled = false;
    gain.addEventListener('input',update);
    update();
  }
})();

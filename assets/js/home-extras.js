/**
 * Home Page Extras
 * 
 * Home-specific behavior that can't be handled by CSS alone.
 * The description positioning requires comparing element heights.
 */
(function() {
  'use strict';

  const header = document.getElementById('about');
  const desc = document.getElementById('description');
  const dummyDesc = document.getElementById('dummy-description');
  const info = document.getElementById('my-infobox');

  if (!header || !desc || !info) return;

  let descHeightVmin = CollapsibleHeader.pxToVmin(desc.offsetHeight);

  header.addEventListener('header:update', (e) => {
    const heightVmin = e.detail.currentHeightVmin;
    
    if (descHeightVmin > heightVmin) {
      if (!info.classList.contains('scrolled')) {
        if (dummyDesc) dummyDesc.style.width = desc.offsetWidth + 'px';
        desc.style.width = desc.offsetWidth + 'px';
      }
      info.classList.add('scrolled');
    } else {
      info.classList.remove('scrolled');
      desc.style.width = 'auto';
    }
  });

  window.addEventListener('resize', () => {
    descHeightVmin = CollapsibleHeader.pxToVmin(desc.offsetHeight);
  }, { passive: true });
})();

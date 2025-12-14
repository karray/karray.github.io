/**
 * Home Page - HeaderController with home-specific callbacks
 */
(function() {
  'use strict';

  const header = document.getElementById('about');
  if (!header) return;

  // DOM elements for home-specific behavior
  const desc = document.getElementById('description');
  const dummyDesc = document.getElementById('dummy-description');
  const info = document.getElementById('my-infobox');

  // Cache description height for scrolled state detection
  let descHeightVmin = HeaderController.pxToVmin(desc?.offsetHeight || 0);

  // Initialize HeaderController
  HeaderController.init({
    headerSelector: '#about',
    onStateChange: handleStateChange,
    onHeightChange: handleHeightChange
  });

  function handleStateChange(state) {
    // Toggle button styling based on collapsed state
    header.classList.toggle('buttons', !state.isCollapsed);
  }

  function handleHeightChange(heightVmin) {
    // Handle description fixed positioning when scrolling
    if (descHeightVmin > heightVmin) {
      if (!info?.classList.contains('scrolled')) {
        if (dummyDesc) dummyDesc.style.width = desc.offsetWidth + 'px';
        if (desc) desc.style.width = desc.offsetWidth + 'px';
      }
      info?.classList.add('scrolled');
    } else {
      info?.classList.remove('scrolled');
      if (desc) desc.style.width = 'auto';
    }
  }

  // Handle resize - update cached description height
  window.addEventListener('resize', () => {
    descHeightVmin = HeaderController.pxToVmin(desc?.offsetHeight || 0);
  }, { passive: true });
})();
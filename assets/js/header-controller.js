/**
 * HeaderController - Unified collapsible header module
 * 
 * Uses vmin units for smooth, viewport-relative header behavior.
 * Configuration-driven, no hardcoded variants.
 */
const HeaderController = (function() {
  'use strict';

  // Shared utility: Convert px to vmin
  function pxToVmin(px) {
    const minDimension = Math.min(window.innerHeight, window.innerWidth);
    return (px / minDimension) * 100;
  }

  // Configuration defaults
  const defaults = {
    headerSelector: null,        // Required: CSS selector for header element
    fullHeightVmin: 60,          // Initial header height in vmin
    collapseThresholdVmin: 15,   // When to switch to fixed mode (vmin)
    enableHideOnScroll: false,   // Hide header when scrolling down
    hideThresholdVmin: 10,       // Additional scroll past collapse before hiding
    onStateChange: null,         // Callback: (state) => void
    onHeightChange: null         // Callback: (heightVmin) => void
  };

  let config = {};
  let headerEl = null;
  let state = {
    isCollapsed: false,
    isHidden: false,
    currentHeightVmin: 0,
    lastScrollVmin: 0,
    scrollDirection: 'down'
  };

  // Get current scroll position in vmin
  function getScrollVmin() {
    const scrollPx = window.scrollY || document.documentElement.scrollTop || 0;
    return pxToVmin(scrollPx);
  }

  // Update scroll direction tracking
  function updateScrollDirection(currentScrollVmin) {
    const threshold = 0.5; // vmin threshold for direction change
    if (currentScrollVmin > state.lastScrollVmin + threshold) {
      state.scrollDirection = 'down';
    } else if (currentScrollVmin < state.lastScrollVmin - threshold) {
      state.scrollDirection = 'up';
    }
    state.lastScrollVmin = currentScrollVmin;
  }

  // Core update function - called on scroll/resize
  function update() {
    if (!headerEl) return;

    const scrollVmin = getScrollVmin();
    const newHeightVmin = config.fullHeightVmin - scrollVmin;

    state.currentHeightVmin = newHeightVmin;

    // Apply height using vmin units
    headerEl.style.height = newHeightVmin + 'vmin';

    // Determine collapsed state
    const wasCollapsed = state.isCollapsed;
    state.isCollapsed = newHeightVmin < config.collapseThresholdVmin;

    // Toggle collapsed class
    if (state.isCollapsed) {
      headerEl.classList.add('fixed-header');
    } else {
      headerEl.classList.remove('fixed-header');
      headerEl.classList.remove('hidden-header');
      state.isHidden = false;
    }

    // Handle hide-on-scroll
    if (config.enableHideOnScroll && state.isCollapsed) {
      updateScrollDirection(scrollVmin);

      const hideThreshold = config.fullHeightVmin + config.hideThresholdVmin;
      const isPastHideThreshold = scrollVmin > hideThreshold;
      const wasHidden = state.isHidden;

      if (state.scrollDirection === 'down' && isPastHideThreshold) {
        state.isHidden = true;
        headerEl.classList.add('hidden-header');
      } else if (state.scrollDirection === 'up') {
        state.isHidden = false;
        headerEl.classList.remove('hidden-header');
      }

      if (wasHidden !== state.isHidden && config.onStateChange) {
        config.onStateChange({ ...state });
      }
    }

    // Fire callbacks
    if (wasCollapsed !== state.isCollapsed && config.onStateChange) {
      config.onStateChange({ ...state });
    }

    if (config.onHeightChange) {
      config.onHeightChange(newHeightVmin);
    }
  }

  // Throttled update using requestAnimationFrame
  let rafPending = false;
  function requestUpdate() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      update();
      rafPending = false;
    });
  }

  // Initialize the controller
  function init(options) {
    config = { ...defaults, ...options };

    if (!config.headerSelector) {
      console.error('HeaderController: headerSelector is required');
      return false;
    }

    headerEl = document.querySelector(config.headerSelector);
    if (!headerEl) {
      console.error(`HeaderController: element not found: ${config.headerSelector}`);
      return false;
    }

    // Set up event listeners
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate, { passive: true });

    // Initial update
    update();

    return true;
  }

  // Destroy the controller
  function destroy() {
    window.removeEventListener('scroll', requestUpdate);
    window.removeEventListener('resize', requestUpdate);
    headerEl = null;
    config = { ...defaults };
    state = {
      isCollapsed: false,
      isHidden: false,
      currentHeightVmin: 0,
      lastScrollVmin: 0,
      scrollDirection: 'down'
    };
  }

  // Get current state
  function getState() {
    return { ...state };
  }

  // Public API
  return {
    init,
    destroy,
    getState,
    update: requestUpdate,
    pxToVmin  // Expose utility for use by page scripts
  };
})();

// Export for module systems (if available)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HeaderController;
}

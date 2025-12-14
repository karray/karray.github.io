/**
 * HeaderController Plugin
 * 
 * Auto-initializing collapsible header with data-attribute configuration.
 * Add `data-collapsible-header` to any header element to enable.
 * 
 * Data Attributes:
 *   data-collapsible-header  - Enable the plugin (required)
 *   data-full-height         - Initial height in vmin (default: 60)
 *   data-collapse-threshold  - Collapse trigger in vmin (default: 15)
 *   data-hide-on-scroll      - Hide header when scrolling down (default: false)
 *   data-hide-threshold      - Extra scroll before hiding in vmin (default: 10)
 * 
 * Events (dispatched on the header element):
 *   header:collapse   - Fired when header becomes collapsed
 *   header:expand     - Fired when header expands
 *   header:hide       - Fired when header hides (scroll down)
 *   header:show       - Fired when header shows (scroll up)
 *   header:update     - Fired on every scroll/resize with { heightVmin, isCollapsed, isHidden }
 */
(function() {
  'use strict';

  // Default configuration
  const defaults = {
    fullHeightVmin: 60,
    collapseThresholdVmin: 15,
    hideOnScroll: false,
    hideThresholdVmin: 10
  };

  // Utility: Convert px to vmin
  function pxToVmin(px) {
    const minDimension = Math.min(window.innerHeight, window.innerWidth);
    return (px / minDimension) * 100;
  }

  // Get scroll position in vmin
  function getScrollVmin() {
    const scrollPx = window.scrollY || document.documentElement.scrollTop || 0;
    return pxToVmin(scrollPx);
  }

  // Dispatch custom event on element
  function emit(element, eventName, detail = {}) {
    element.dispatchEvent(new CustomEvent(eventName, { 
      bubbles: true, 
      detail 
    }));
  }

  // Header instance class
  class CollapsibleHeader {
    constructor(element) {
      this.el = element;
      this.config = this.parseConfig();
      this.state = {
        isCollapsed: false,
        isHidden: false,
        currentHeightVmin: this.config.fullHeightVmin,
        lastScrollVmin: 0,
        scrollDirection: 'down'
      };
      this.rafPending = false;
      
      this.bindEvents();
      this.update();
    }

    parseConfig() {
      const el = this.el;
      return {
        fullHeightVmin: parseFloat(el.dataset.fullHeight) || defaults.fullHeightVmin,
        collapseThresholdVmin: parseFloat(el.dataset.collapseThreshold) || defaults.collapseThresholdVmin,
        hideOnScroll: el.dataset.hideOnScroll === 'true',
        hideThresholdVmin: parseFloat(el.dataset.hideThreshold) || defaults.hideThresholdVmin
      };
    }

    bindEvents() {
      this.handleScroll = () => this.requestUpdate();
      this.handleResize = () => this.requestUpdate();
      
      window.addEventListener('scroll', this.handleScroll, { passive: true });
      window.addEventListener('resize', this.handleResize, { passive: true });
    }

    requestUpdate() {
      if (this.rafPending) return;
      this.rafPending = true;
      requestAnimationFrame(() => {
        this.update();
        this.rafPending = false;
      });
    }

    updateScrollDirection(scrollVmin) {
      const threshold = 0.5;
      if (scrollVmin > this.state.lastScrollVmin + threshold) {
        this.state.scrollDirection = 'down';
      } else if (scrollVmin < this.state.lastScrollVmin - threshold) {
        this.state.scrollDirection = 'up';
      }
      this.state.lastScrollVmin = scrollVmin;
    }

    update() {
      const scrollVmin = getScrollVmin();
      const newHeightVmin = this.config.fullHeightVmin - scrollVmin;

      this.state.currentHeightVmin = newHeightVmin;

      // Apply height
      this.el.style.height = newHeightVmin + 'vmin';

      // Collapse state
      const wasCollapsed = this.state.isCollapsed;
      this.state.isCollapsed = newHeightVmin < this.config.collapseThresholdVmin;

      if (this.state.isCollapsed) {
        this.el.classList.add('fixed-header');
      } else {
        this.el.classList.remove('fixed-header');
        this.el.classList.remove('hidden-header');
        this.state.isHidden = false;
      }

      // Emit collapse/expand events
      if (wasCollapsed !== this.state.isCollapsed) {
        emit(this.el, this.state.isCollapsed ? 'header:collapse' : 'header:expand', { ...this.state });
      }

      // Hide on scroll
      if (this.config.hideOnScroll && this.state.isCollapsed) {
        this.updateScrollDirection(scrollVmin);

        const hideThreshold = this.config.fullHeightVmin + this.config.hideThresholdVmin;
        const isPastHideThreshold = scrollVmin > hideThreshold;
        const wasHidden = this.state.isHidden;

        if (this.state.scrollDirection === 'down' && isPastHideThreshold) {
          this.state.isHidden = true;
          this.el.classList.add('hidden-header');
        } else if (this.state.scrollDirection === 'up') {
          this.state.isHidden = false;
          this.el.classList.remove('hidden-header');
        }

        // Emit hide/show events
        if (wasHidden !== this.state.isHidden) {
          emit(this.el, this.state.isHidden ? 'header:hide' : 'header:show', { ...this.state });
        }
      }

      // Always emit update event
      emit(this.el, 'header:update', { ...this.state });
    }

    destroy() {
      window.removeEventListener('scroll', this.handleScroll);
      window.removeEventListener('resize', this.handleResize);
    }
  }

  // Store instances
  const instances = new WeakMap();

  // Auto-init on DOMContentLoaded
  function init() {
    document.querySelectorAll('[data-collapsible-header]').forEach(el => {
      if (!instances.has(el)) {
        instances.set(el, new CollapsibleHeader(el));
      }
    });
  }

  // Public API
  window.CollapsibleHeader = {
    init,
    getInstance: (el) => instances.get(el),
    pxToVmin
  };

  // Auto-init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

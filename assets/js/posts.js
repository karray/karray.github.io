/**
 * Post Page - HeaderController with hide-on-scroll enabled
 */
(function() {
  'use strict';

  const header = document.getElementById('post-header');
  if (!header) return;

  HeaderController.init({
    headerSelector: '#post-header',
    enableHideOnScroll: true
  });
})();
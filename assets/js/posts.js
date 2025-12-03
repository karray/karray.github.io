(function() {
    const header = document.getElementById('post-header');
    if (!header) return;

    const HEADER_HEIGHT_VMIN = 60;
    const UNIT = 'vmin';
    const THRESHOLD = 15;
    
    let prevScroll = 0;

    function getMinViewHeightWidth(x) {
        return x / Math.min(window.innerHeight, window.innerWidth) * 100;
    }

    function getScroll() {
        return getMinViewHeightWidth(window.scrollY || document.documentElement.scrollTop);
    }

    function updateElements() {
        const scroll = getScroll();
        const newHeight = HEADER_HEIGHT_VMIN - scroll;
        
        header.style.height = `${newHeight}${UNIT}`;

        if (newHeight < THRESHOLD) {
            header.classList.add('fixed-header');
            
            // Hide the header if scrolling down and past the threshold
            if (newHeight < -THRESHOLD && scroll > prevScroll) {
                header.classList.add('hidden-header');
            } else {
                header.classList.remove('hidden-header');
            }
            prevScroll = scroll;
        } else {
            header.classList.remove('fixed-header');
        }
    }

    window.addEventListener('scroll', () => requestAnimationFrame(updateElements));
    window.addEventListener('resize', updateElements);
    
    // Initial call
    updateElements();
})();
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

    function setupCodeBlocks() {
        const codeBlocks = document.querySelectorAll('.highlighter-rouge');
        
        codeBlocks.forEach(block => {
            // Find the code element
            const codeElement = block.querySelector('code');
            if (!codeElement) return;
            
            // Get language class
            // Check both the code element and the wrapper block
            // Jekyll often puts 'language-xyz' on the .highlighter-rouge wrapper
            const allClasses = [
                ...Array.from(block.classList),
                ...Array.from(codeElement.classList)
            ];
            
            let language = '';
            
            // Try to find language class
            const langClass = allClasses.find(c => c.startsWith('language-'));
            if (langClass) {
                language = langClass.replace('language-', '');
            } else {
                // Fallback: try to detect from data attributes or default to 'Code'
                language = block.getAttribute('data-lang') || 'Code';
            }

            // Create header container
            const header = document.createElement('div');
            header.className = 'code-header';
            
            // Create language label
            const label = document.createElement('span');
            label.className = 'language-label';
            label.textContent = language;
            
            // Create copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-button';
            copyBtn.setAttribute('aria-label', 'Copy code');
            copyBtn.innerHTML = `
                <svg viewBox="0 0 24 24">
                    <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                </svg>
                <span>Copy</span>
            `;
            
            // Add copy functionality
            copyBtn.addEventListener('click', async () => {
                try {
                    // Get text content of the code block
                    const code = block.querySelector('code').innerText;
                    await navigator.clipboard.writeText(code);
                    
                    // Success feedback
                    copyBtn.classList.add('copied');
                    copyBtn.innerHTML = `
                        <svg viewBox="0 0 24 24">
                            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                        </svg>
                        <span>Copied!</span>
                    `;
                    
                    // Reset after 2 seconds
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.innerHTML = `
                            <svg viewBox="0 0 24 24">
                                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                            </svg>
                            <span>Copy</span>
                        `;
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    copyBtn.classList.add('error');
                    copyBtn.querySelector('span').textContent = 'Error';
                    setTimeout(() => {
                        copyBtn.classList.remove('error');
                        copyBtn.querySelector('span').textContent = 'Copy';
                    }, 2000);
                }
            });
            
            header.appendChild(label);
            header.appendChild(copyBtn);
            block.appendChild(header);
        });
    }

    // Initialize code blocks
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupCodeBlocks);
    } else {
        setupCodeBlocks();
    }
})();
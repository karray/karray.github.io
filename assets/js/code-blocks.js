/**
 * Code Blocks Enhancement Module
 * 
 * Adds language labels and copy buttons to code blocks
 */
(function() {
  'use strict';

  function setupCodeBlocks() {
    const codeBlocks = document.querySelectorAll('.highlighter-rouge');
    
    codeBlocks.forEach(block => {
      // Skip if already processed
      if (block.querySelector('.code-header')) return;
      
      // Find the code element
      const codeElement = block.querySelector('code');
      if (!codeElement) return;
      
      // Get language class from wrapper or code element
      const allClasses = [
        ...Array.from(block.classList),
        ...Array.from(codeElement.classList)
      ];
      
      let language = '';
      const langClass = allClasses.find(c => c.startsWith('language-'));
      if (langClass) {
        language = langClass.replace('language-', '');
      } else {
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

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupCodeBlocks);
  } else {
    setupCodeBlocks();
  }
})();

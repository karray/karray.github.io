// --- 1. THEME LOGIC ---
const themeBtn = document.getElementById('theme-toggle-button');
const currentTheme = localStorage.getItem('theme');

// Initial Load
if (currentTheme) {
    document.body.className = currentTheme;
} else {
    // System preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.className = 'dark-mode';
    }
}

// Toggle Event
themeBtn.addEventListener('click', function () {
    if (document.body.classList.contains('dark-mode')) {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('theme', 'light-mode');
    } else {
        document.body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark-mode');
    }
});

// --- 2. TUTORIAL LOGIC (SIDE POSITIONING) ---
const tutorialOverlay = document.getElementById('tutorial-overlay');
const spotlight = document.getElementById('tutorial-spotlight');
const card = document.getElementById('tutorial-card');

const titleEl = document.getElementById('tut-title');
const descEl = document.getElementById('tut-desc');
const nextBtn = document.getElementById('tut-next');
const skipBtn = document.getElementById('tut-skip');
const stepEl = document.getElementById('tut-step');
const totalEl = document.getElementById('tut-total');

const steps = [
    { target: '#control-panel', title: "Control Center", desc: "Manage the camera, upload images, change mode, or adjust settings here. You can fine-tune the visualization mode and opacity." },
    { target: '#upload-group', title: "Upload & Presets", desc: "Upload your own image or choose from our preset scenarios" },
    { target: '#mode-select', title: "Model Modes", desc: "Select different model modes: default ResNet with label-free/Grad-CAM heatmaps, Multi-instance classification, object detection" },
    { target: '#predictions-panel', title: "Predictions", desc: "The model's top guesses appear here. Click a prediction to highlight the area in the image responsible for that classification. You can clear selection by clicking again or clicking X-button." },
    { target: '#canvas-area', title: "Live Visualization", desc: "This canvas displays the model's salience map. Pause the stream to interact with the overlay Grid and explore specific regions. You can click on grid cells to show predictions for that region (multiple cells selection possible)." },
];

let currentStepIndex = 0;

function updateSpotlightPosition() {
    // Update overlay height for scrolling
    tutorialOverlay.style.height = document.documentElement.scrollHeight + 'px';

    const step = steps[currentStepIndex];
    const targetEl = document.querySelector(step.target);
    if (!targetEl) return;

    const rect = targetEl.getBoundingClientRect(); // Viewport relative
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
    const padding = 6;

    // Absolute Coords
    const absTop = rect.top + scrollTop - padding;
    const absLeft = rect.left + scrollLeft - padding;
    const width = rect.width + (padding * 2);
    const height = rect.height + (padding * 2);

    // Set Spotlight
    spotlight.style.width = width + 'px';
    spotlight.style.height = height + 'px';
    spotlight.style.top = absTop + 'px';
    spotlight.style.left = absLeft + 'px';

    // --- SMART POSITIONING ---
    // We want to place the card to the RIGHT if there is space, otherwise Bottom, etc.

    const cardWidth = 340; // Approx card width
    const cardGap = 20;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    // Check Available Space relative to viewport
    const spaceRight = windowWidth - rect.right;
    const spaceLeft = rect.left;
    const spaceBottom = windowHeight - rect.bottom;

    // Logic:
    // 1. If mobile (<900px), use CSS Fixed positioning (do nothing here regarding top/left).
    // 2. If space on RIGHT > cardWidth, place Right.
    // 3. Else if space on LEFT > cardWidth, place Left.
    // 4. Else place Bottom.
    // 5. If not enough space Bottom, place Top.

    if (windowWidth < 900) {
        // Mobile: Reset inline styles so CSS handles it
        card.style.top = '';
        card.style.left = '';
        card.style.transform = '';
    } else {
        // Desktop
        let finalLeft, finalTop;

        // Priority 1: Right Side
        if (spaceRight > (cardWidth + cardGap)) {
            finalLeft = absLeft + width + cardGap;
            finalTop = absTop; // Align Top edges
        }
        // Priority 2: Left Side
        else if (spaceLeft > (cardWidth + cardGap)) {
            finalLeft = absLeft - cardWidth - cardGap;
            finalTop = absTop;
        }
        // Priority 3: Bottom
        else {
            finalLeft = absLeft; // Align Left edges
            if (spaceBottom > 250) {
                finalTop = absTop + height + cardGap; // Below
            } else {
                finalTop = absTop - 250; // Above (approx height)
            }
        }

        // Boundaries Check (keep inside document)
        if (finalLeft + cardWidth > document.documentElement.scrollWidth) {
            finalLeft = document.documentElement.scrollWidth - cardWidth - 10;
        }
        if (finalLeft < 10) finalLeft = 10;

        card.style.left = finalLeft + 'px';
        card.style.top = finalTop + 'px';
    }

    card.classList.add('visible');
}

function setStepContent() {
    const step = steps[currentStepIndex];
    titleEl.innerText = step.title;
    descEl.innerText = step.desc;
    stepEl.innerText = currentStepIndex + 1;
    totalEl.innerText = steps.length;
    nextBtn.innerText = (currentStepIndex === steps.length - 1) ? "Finish" : "Next";
}

function goToStep(index) {
    currentStepIndex = index;
    setStepContent();

    const targetEl = document.querySelector(steps[index].target);
    targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Delay to allow scroll to settle
    setTimeout(updateSpotlightPosition, 150);
    setTimeout(updateSpotlightPosition, 400);
}

function startTutorial() {
    currentStepIndex = 0;
    tutorialOverlay.classList.add('active');
    goToStep(0);
}
function endTutorial() {
    card.classList.remove('visible');
    setTimeout(() => { tutorialOverlay.classList.remove('active'); }, 300);
    localStorage.setItem('lafam_tutorial_shown', 'true');
}

nextBtn.addEventListener('click', () => {
    if (currentStepIndex < steps.length - 1) goToStep(currentStepIndex + 1);
    else endTutorial();
});
skipBtn.addEventListener('click', endTutorial);
document.getElementById('help-button').addEventListener('click', startTutorial);

window.addEventListener('resize', () => {
    if (tutorialOverlay.classList.contains('active')) updateSpotlightPosition();
});

window.addEventListener('load', () => {

    if (!localStorage.getItem('lafam_tutorial_shown')) setTimeout(startTutorial, 1000);
});

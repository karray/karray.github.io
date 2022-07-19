function getScroll() {
    return getMinViewHeightWidth(window.pageYOffset || document.documentElement.scrollTop);
}

function getMinViewHeightWidth(x){
    return x/Math.min(window.innerHeight, window.innerWidth)*100;
}

let header = document.getElementById('post-header');
const unit = 'vmin';

const header_height = 60;

let is_small_screen = window.matchMedia('(max-width: 350px)')
let prevScroll = 0

function update_elements(){
    if(is_small_screen.matches){
        about.classList.add('fixed-header')
        return
    }

    let scroll = getScroll()
    let new_height = header_height - scroll
    header.style.height = new_height + unit

    if(new_height<15) {
        header.classList.add('fixed-header')
        // hide the header if scrolling down
        if(new_height < -15 && scroll > prevScroll) {
            header.classList.add('hidden-header')
        }
        else {
            header.classList.remove('hidden-header')
        }
        prevScroll = scroll
    }
    else {
        header.classList.remove('fixed-header')
    }
}

window.addEventListener('scroll', ()=>requestAnimationFrame(update_elements));
window.addEventListener('resize', function(event) {
    update_elements()
}, true);

update_elements()
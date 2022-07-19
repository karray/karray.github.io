function getMinViewport(){
    if(window.innerHeight > window.innerWidth){
        return 'vw';
    }
    return 'vh';
}
function getScroll() {
    return getMinViewHeightWidth(window.pageYOffset || document.documentElement.scrollTop);
}

function getMinViewHeightWidth(x){
    return x/Math.min(window.innerHeight, window.innerWidth)*100;
}

let header = document.getElementById('post-header');
let unit = getMinViewport();

let header_height = 60;

let is_small_screen = window.matchMedia('(max-width: 350px)')

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
    }
    else {
        header.classList.remove('fixed-header')
    }
}

window.addEventListener('scroll', ()=>requestAnimationFrame(update_elements));
window.addEventListener('resize', function(event) {
    requestAnimationFrame(update_elements)
}, true);

requestAnimationFrame(update_elements)
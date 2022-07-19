function getScroll() {
    return getMinViewHeightWidth(window.pageYOffset || document.documentElement.scrollTop);
}

function getMinViewHeightWidth(x){
    return x/Math.min(window.innerHeight, window.innerWidth)*100;
}

let about = document.getElementById('about');
let me_img = document.getElementById('me');
let desc = document.getElementById('description');
let dummy_desc = document.getElementById('dummy-description');
let info = document.getElementById('my-infobox');
const unit = 'vmin'

const about_height = 60;
let desc_height = getMinViewHeightWidth(desc.offsetHeight)

let is_small_screen = window.matchMedia('(max-width: 350px)')

function update_elements(){
    if(is_small_screen.matches){
        about.classList.add('fixed-header')
        return
    }

    let scroll = getScroll()
    let new_height = about_height - scroll
    about.style.height = new_height + unit
    if(desc_height > new_height){
        if(!info.classList.contains('scrolled')){
            dummy_desc.style.width = desc.offsetWidth+'px';
            desc.style.width = desc.offsetWidth+'px';
        }
        info.classList.add('scrolled')
    }
    else{
        info.classList.remove('scrolled')
        desc.style.width = 'auto';
    }

    if(new_height<15) {
        about.classList.add('fixed-header')
    }
    else {
        about.classList.remove('fixed-header')
    }
}

window.addEventListener('scroll', ()=>requestAnimationFrame(update_elements));
window.addEventListener('resize', function(event) {
    desc_height = getMinViewHeightWidth(desc.offsetHeight)
    unit = getMinViewport();
    update_elements()
}, true);

// ['#FFC0CB', '#DB7093', '#E6E6FA', '#D8BFD8', '#DDA0DD', '#BA55D3', '#7B68EE', '#663399', '#FFA07A', '#FA8072', '#FF8C00',
// '#FF6347', '#90EE90', '#3CB371', '#228B22', '	#9ACD32', '#66CDAA', '#20B2AA', '#008080', '#00CED1', '#6495ED']

update_elements()
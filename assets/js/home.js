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

let about = document.getElementById('about');
let me_img = document.getElementById('me');
let desc = document.getElementById('description');
let dummy_desc = document.getElementById('dummy-description');
let info = document.getElementById('my-infobox');
let unit = getMinViewport();

let about_height = 60;
// console.log(getMinViewHeightWidth(60))
let desc_height = getMinViewHeightWidth(desc.offsetHeight)
// let me_width = to_vh(me_img.offsetWidth);

let is_small_screen = window.matchMedia('(max-width: 350px)')

function update_elements(){
    if(is_small_screen.matches){
        about.classList.add('fixed-header')
        // me_img.style.width = '100%'
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
        // desc.classList.remove('fixed')
    }
    else{
        info.classList.remove('scrolled')
        desc.style.width = 'auto';

        // desc.classList.add('fixed')
    }

    if(new_height<15) {
        about.classList.add('fixed-header')
        // me_img.style.width = '100%'
    }
    else {
        about.classList.remove('fixed-header')
        // me_img.style.width = me_width - scroll + 'vh';
        // me_img.style.top = (60-me_width)/2 + scroll + 'vh';
    }
}

window.addEventListener('scroll', update_elements);
window.addEventListener('resize', function(event) {
    // me_width = to_vh(me_img.offsetWidth);
    // about_height = getMaxViewHeightWidth(60);
    desc_height = getMinViewHeightWidth(desc.offsetHeight)
    unit = getMinViewport();
    // is_small_screen = window.matchMedia(`(max-width: ${breakpoint})`).matches
    update_elements()
}, true);

// ['#FFC0CB', '#DB7093', '#E6E6FA', '#D8BFD8', '#DDA0DD', '#BA55D3', '#7B68EE', '#663399', '#FFA07A', '#FA8072', '#FF8C00',
// '#FF6347', '#90EE90', '#3CB371', '#228B22', '	#9ACD32', '#66CDAA', '#20B2AA', '#008080', '#00CED1', '#6495ED']

update_elements()
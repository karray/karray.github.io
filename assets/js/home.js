function to_vh(x){
    return x / window.innerHeight*100;
}
function getScroll() {
    return to_vh(window.pageYOffset || document.documentElement.scrollTop);
}

let about = document.getElementById('about');
let me_img = document.getElementById('me');
let desc = document.getElementById('description');

let about_height = 60 - 15;
let desc_height = to_vh(about.offsetHeight - desc.offsetHeight)
let me_width = 50;

let is_small_screen = window.matchMedia( "(max-width: 1000px)" ).matches

function update_elements(){
    if(is_small_screen){
        me_img.style.width = 'initial'
        return
    }

    let scroll = getScroll()
    me_img.style.width = me_width - scroll + 'vh';
    console.log(me_width, scroll)
    if(desc_height < scroll){
        desc.classList.add('scrolled')
        desc.classList.remove('fixed')
    }
    else{
        desc.classList.remove('scrolled')
        desc.classList.add('fixed')
    }

    if(scroll > about_height)
        about.classList.add('fixed-header')
    else
        about.classList.remove('fixed-header')
    
}

window.addEventListener('scroll', update_elements);
window.addEventListener('resize', function(event) {
    desc_height = to_vh(about.offsetHeight - desc.offsetHeight)
    is_small_screen = window.matchMedia( "(max-width: 1000px)" ).matches
    update_elements()
}, true);

['#FFC0CB', '#DB7093', '#E6E6FA', '#D8BFD8', '#DDA0DD', '#BA55D3', '#7B68EE', '#663399', '#FFA07A', '#FA8072', '#FF8C00',
'#FF6347', '#90EE90', '#3CB371', '#228B22', '	#9ACD32', '#66CDAA', '#20B2AA', '#008080', '#00CED1', '#6495ED']

update_elements()
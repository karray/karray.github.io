function to_vh(x){
    return x / window.innerHeight*100;
}
function getScroll() {
    return to_vh(window.pageYOffset || document.documentElement.scrollTop);
}

let about = document.getElementById('about');
let me = document.getElementById('me');
let desc = document.getElementById('description');

let about_height = 60 - 15;
let desc_height = to_vh(about.offsetHeight - desc.offsetHeight)
let me_width = to_vh(me.offsetHeight);

function update_elements(){
    let scroll = getScroll()
    me.style.width = me_width  - scroll + 'vh';
    if(desc_height < scroll){
        desc.classList.add('scrolled')
        desc.classList.remove('fixed')
    }
    else{
        desc.classList.remove('scrolled')
        desc.classList.add('fixed')
    }

    if(scroll > about_height)
        about.classList.add('hidden')
    else
        about.classList.remove('hidden')
}

window.addEventListener('scroll', update_elements);
window.addEventListener('resize', function(event) {
    desc_height = to_vh(about.offsetHeight - desc.offsetHeight)
    update_elements()
}, true);

update_elements()
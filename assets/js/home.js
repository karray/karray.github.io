function to_vh(x){
    return x / window.innerHeight*100;
}
function getScroll() {
    return to_vh(window.pageYOffset || document.documentElement.scrollTop);
}

let about = document.getElementById('about');
let me = document.getElementById('me');
let desc = document.getElementById('description');
let vh5 = to_vh(5);
let desc_height = to_vh(about.offsetHeight - desc.offsetHeight) - vh5
let me_width = to_vh(me.offsetHeight);

window.addEventListener('scroll', function () {
    let scroll = getScroll()
    me.style.width = me_width  - scroll + 'vh';
    console.log(getScroll(), to_vh(about.offsetHeight - desc.offsetHeight));
    if(desc_height < getScroll()){
        desc.classList.add('scrolled')
        desc.classList.remove('fixed')
    }
    else{
        desc.classList.remove('scrolled')
        desc.classList.add('fixed')
    }
});

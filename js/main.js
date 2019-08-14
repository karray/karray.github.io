$(function () {
    var sticky = $("#sticky");
    var parallax = $('.parallax');
    var not_sticky=$("#not-sticky");
    var me=$("#me");
    var me_width = me.width();

    var offset = not_sticky.offset().top;

    parallax.on('scroll', function () {
        var current_offset = not_sticky.offset().top;
        // console.log(current_offset-offset, me.width(), me_width)
        if (current_offset <= 0) {
            sticky.addClass("scrolled");
        } else {
            me.width(me_width+current_offset-offset);
            sticky.removeClass("scrolled");
        }
    });
})
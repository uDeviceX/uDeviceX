#!/usr/bin/awk -f
function vor(ax, ay, bx, by, cx, cy) {
    bx -= ax; by -= ay
    cx -= ax; cy -= ay

    x = -(by*cy^2-by^2*cy-bx^2*cy+by*cx^2)/(2*(bx*cy-by*cx))
    y = (bx*cy^2+bx*cx^2-by^2*cx-bx^2*cx)/(2*(bx*cy-by*cx))

    x += ax; y += ay
}

BEGIN {
    ax = ARGV[1]; shift()
    ay = ARGV[1]; shift()

    bx = ARGV[1]; shift()
    by = ARGV[1]; shift()

    cx = ARGV[1]; shift()
    cy = ARGV[1]; shift()

    vor(ax, ay, bx, by, cx, cy)

    print ax, ay
    print bx, by
    print cx, cy
    print ax, ay
    printf "\n\n"
    print x, y
}

function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }

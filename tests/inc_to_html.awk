#!/usr/bin/awk -f

# syntax: #include test_data/test.html

BEGIN {
    dir = "html"
    idx = dir "/index.html"
    w = 500; h = 500
    img_templ = "<img src=\"%s\" width=\"%d\" height=\"%d\">"
    img_templ = "<p>" img_templ "</p>"
    img_templ = "\n" img_templ "\n"

    system("mkdir -p " dir)
}

$1 ~ /^#include/ {
    d = $2
    id = $2
    desc = d "/desc.html"
    imgA0 = "img1.png"; imgB0 = "img2.png"
    imgA1 = rename_img(imgA0); imgB1 = rename_img(imgB0)
    copy_img(imgA0, imgA1); copy_img(imgB0, imgB1)

    read_file(desc)

    gen_img_desc(imgA1); gen_img_desc(imgB1)

    next
}

{
    print > idx
}

function rename_img(img) {
    return id "." img
}

function copy_img(img0, img1,  cmd) {
    cmd = sprintf("cp %s/%s %s/%s", d, img0, dir, img1)
    print "CMD " cmd
    system(cmd)
}

function read_file(file,  x) {
    while (getline x <file > 0) print x > idx
    close(file)
}

function gen_img_desc(img) {
    printf img_templ, img, w, h > idx
}

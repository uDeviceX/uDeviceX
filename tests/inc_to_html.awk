#!/usr/bin/awk -f

# syntax: #include test_data/test.html

BEGIN {
    dir = "html"
    idx = dir "/README.md"
    w = 500; h = 500
    img_templ = "<img src=\"%s\" width=\"%d\" height=\"%d\">"
    img_templ = "<p>" img_templ "</p>"
    img_templ = "\n" img_templ "\n"
    s = "success.txt"

    system("mkdir -p " dir)
}

$1 ~ /^#include/ {
    d = $2
    print "Processing test " d

    # generate test description
    desc = d "/desc.html"
    read_file(desc)

    # check the status and react accordingly
    if (exists(s)) print "![alt text](success.jpg =50x50)" > idx
    else           print "![alt text](fail.jpg =50x50)"    > idx

    # # always shows images
    # for (id = 1; 1; id++) {
    #     img = sprintf("img%d.png", id)
    #     if (exists(img)) process_img(img)
    #     else             break
    # }

    # ![alt text](img2.png "Logo Title Text 1")

    next
}

{
    print > idx
}

function process_img(img0) {
    # rename
    img1 = d "." img0
    
    # copy
    cmd = sprintf("cp %s/%s %s/%s", d, img0, dir, img1)
    system(cmd)

    # generate description
    printf img_templ, img1, w, h > idx
}

function read_file(file,  x) {
    while (getline x <file > 0) print x > idx
    close(file)
}

function exists(f) {
    cmd = sprintf("test -f %s/%s", d, f)
    return !system(cmd)
}

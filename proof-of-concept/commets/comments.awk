#!/usr/bin/awk -f

function err(s) {
    printf "(comments.awk) %s\n", s
    exit
}

function logg(s,    cmd) {
    cmd = "cat 1>&2"
    printf "(comments.awk) %s\n", s | cmd
    close(cmd)
}

function read_file(fn,    line, sep) { # reads file content into string `a'
    while (getline line < fn > 0) {
	a = a sep line
	sep = "\n"
    }
    if (line != sep) a = a sep
}


function write_file() {
    printf "%s", a
}

function ss(i, l) { # substring of `a' starting at `i' with length `l'
    return substr(a, i, l)
}

function ch(i) { # character `i' in `a'
    return ss(i, 1)
}

function spacep(c) {
    return c == " " || c == "\n" || c == "\t"
}

function find_comment_block(    i, n) { # sets `lo', `hi' for the
					# comments block
    # find comment start `/*'
    n = length(a); i = 1
    while (spacep(ch(i))) i++; # eat spaces
    if (ss(i, 2) != "/*") return !HAS_COMMENT_BLOCK

    lo = i # start of the comment block

    while (1) {
	i++
	if (i == n)            return !HAS_COMMENT_BLOCK
	if (ss(i, 2) == "*/")  {hi = i + 1; return  HAS_COMMENT_BLOCK}
    }
}

function remove(s, lo, hi,    head, tail)  {
    head = substr(s, 1       , lo - 1)
    tail = substr(s, hi + 1        )
    return head tail
}

function replace(s, lo, hi, t,    head, tail)  {
    head = substr(s, 1       , lo - 1)
    tail = substr(s, hi + 1        )
    return head t tail
}

function end_index(s, t,    i) {
    i = index(s, t)
    if (i == 0) return 0
    return i + length(t)
}

function transform_comment(    lo, hi, be, end) {
    beg = " *  Users are NOT authorized"
    end = "permission from the author of this file."

    lo =     index(cm, beg)
    if (lo == 0)             return 0
    hi = end_index(cm, end)
    if (hi == 0)             err("I found beg but cannot find end in comments block in " fn)

    cm = remove(cm, lo, hi)
    return 1
}

function transfrom(    rc) {
    if (!find_comment_block()) return

    cm = substr(a, lo, hi - lo + 1) # get comment block as a string
    rc = transform_comment()        # transfrom comment block

    if (!rc) return

    logg("trans: " fn)
    a = replace(a, lo, hi, cm)
}

function read_cpy(fn,    line, sep) { # reads file content into string `a'
    while (getline line < fn > 0) {
	cpy = cpy sep line
	sep = "\n"
    }
    if (line != sep) cpy = cpy sep
}

BEGIN {
    HAS_COMMENT_BLOCK = 1
    #read_cpy(fcpy = ARGV[1])

    read_file(fn = ARGV[1])
    transfrom()
    write_file()
}

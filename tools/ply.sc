#!/usr/bin/awk -f

function usg(s) {
    msg("usage: ply.sc <scale> <ply.in>     >      <ply.out>")
    exit
};

BEGIN {
    s = ARGV[1]; shift()           # scale
    if (!floatp(s)) { msg("ply.sc: should be a number"); usg() }

    f  = ARGC > 1 ? ARGV[1] : "-"
    read_header(f)
    read_vert(f)
    read_rest(f)
}

function read_header(f) {
    while (getline < f > 0) {
	print
	if ($0 == "end_header") break
    }
}

function read_vert(f,   rc) {
    while (rc = getline < f > 0) {
	if (NF != 3) break
	read_vert0()
    }
    if (rc) print
}

function read_vert0(   x, y, z) {
    x = $1; y = $2; z = $3
    print s*x, s*y, s*z
}

function read_rest(f) {
    while (getline < f > 0) print
}

function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }
function floatp(x)  { return x == x + 0 }
function msg(s)     { printf "%s\n", s | "cat >&2" }

# TEST: ply.sc.t0
# ply.sc 2 test_data/sphere.ply  > sc.out.ply
#

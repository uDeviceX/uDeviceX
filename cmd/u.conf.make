#!/usr/bin/awk -f

BEGIN {
    S = ARGV[1]; shift()
    c = ARGV[1]; shift()
    msg S " " c
}

function msg(s) { printf "%s\n", s | "cat >&2" }
function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }

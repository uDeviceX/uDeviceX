#!/usr/bin/awk -f

BEGIN {
    ERR_NO_RUNS = 3

    S = ARGV[1]; shift()
    c = ARGV[1]; shift()
    nbin = 0 # bin dirs
    parse()
    print "nbin:", nbin
    bins()
}

function bins() {
    if      (nbin == 0) exit ERR_NO_RUNS
    else if (nbin == 1) bins1()
    else                binsN()
}

function bins1(   s, rc) {
    s = " " # space
    rc = cmd("u.conf.make0" s qq(S) s "bin")
}

function binsN(i,  s, rc) {
    for (i = 1; i <= nbin; i++) {
	s = " " # space
	rc = cmd("u.conf.make0" s qq(S) s "bin" "." i)
    }
}

function cmd(s) { return system(s) } # run shell command

function parse() {
    while (getline < c > 0) {
	comm() # stip comments
	if (emptyp()) continue
	if (runp())   nbin += 1
    }
    close(c)
}

function runp() { return $1 == "run" } # run predicate
function comm() { sub(/#.*/, "") }
function emptyp() { return $0 ~ /^[ \t]*$/ }
function qq(s)    { return "'"  s  "'" } # quotes a string

function msg(s) { printf "%s\n", s | "cat >&2" }
function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }

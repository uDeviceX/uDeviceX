#!/usr/bin/awk -f

BEGIN {
    ERR_NO_RUNS = 3

    S = ARGV[1]; shift()
    c = ARGV[1]; shift()
    nbin = 0 # bin dirs
    parse()
    print "nbin:", nbin
    bins()
    if (nbin != 0) make("makefile")
}

function make(f,   i) { # makefile in .
    printf "%s", "all:"                                             > f
    for (i = 1; i <= nbin; i++) printf " bin.%d/udx", i             > f
    printf "\n"                                                     > f

    printf "%s", "clean:"                                           > f
    for (i = 1; i <= nbin; i++) printf " clean.%d", i               > f
    
    printf "\n\n"                                                   > f
    for (i = 1; i <= nbin; i++)
    	printf "bin.%d/udx:; u.make -f bin.%d/Makefile\n", i, i     > f
    printf "\n"                                                     > f
    for (i = 1; i <= nbin; i++)
     	printf "clean.%d:; u.make -f bin.%d/Makefile clean\n", i, i > f
    close(f)
}

function bins() {
    if      (nbin == 0) exit ERR_NO_RUNS
    else if (nbin == 1) bins1()
    else                binsN()
}

function bins1(   sp, b, rc) {
    sp = " " # space
    b = "."
    cmd("u.conf.make0" sp qq(S) sp b)
}

function binsN(i,  sp, b, rc) {
    for (i = 1; i <= nbin; i++) {
	sp = " " # space
	b  = "bin" "." i
	cmd("u.conf.make0" sp qq(S) sp b)
    }
}

function cmd(s,  rc) {
    rc = system(s)
    if (rc != 0) exit rc
}

function parse() {
    while (getline < c > 0) {
	comm() # stip comments
	if (emptyp()) continue
	if (runp())   nbin += 1
    }
    close(c)
}

function runp() { return $1 == "run" } # run predicate
function comm() { sub(/#.*/, "") }     # stip comments
function emptyp() { return $0 ~ /^[ \t]*$/ }
function qq(s)    { return "'"  s  "'" } # quotes a string

function msg(s) { printf "%s\n", s | "cat >&2" }
function shift(  i) { for (i = 2; i < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC-- }

#!/usr/bin/awk -f
#
# Rounds numbers in input
#
# Usage:
# awk '{print $2}' diag.txt | fround.awk
# awk '{print $2}' diag.txt | fround.awk -v tol=6
#

BEGIN {
    if (length(tol) == 0) tol = 3 # default level of tolerance
    fmt =  "%." tol "f"
}

function format(e) {return sprintf(fmt, e)}
function fn(e)     {return e + 0}

function hash(e,  h, ftm) {
    h = format(e)
    if (fn(h) == 0) h = format(0)
    return h
}

{
    l = "" # output line
    for (i = 1; i <= NF; i++) {
	f = hash($i)
	sep = (i == 1) ? "" : " "
	l = l sep f 
    }
    print l
}

#!/usr/bin/awk -f
#
# Round numbers in a first column
#
# Usage:
# awk '{print $2}' diag.txt | fround.awk
# awk '{print $2}' diag.txt | fround.awk -v tol=6
#

BEGIN {
    if (length(tol) == 0) tol = 3 # default level of tolerance
    fmt =  "%." tol "f"
}

function hash(e,  h, ftm) {
    h = sprintf(fmt, e)
    return h
}

{
    print hash($1)
}

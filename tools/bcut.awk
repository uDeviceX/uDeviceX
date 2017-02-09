#!/usr/bin/awk -f

# print only the last block which contains a string :CDFT:

BEGIN {
    pat = ":CDFT:"
}

function good(s) {return index(s, pat)}
function bad(s)  {return !good(s)     }

{
    a[NR] = $0 # put text into arrya
}

function dump_last(    n, hi, lo, k) { # dump only the last block
    n = NR
    for (hi = n    ;  hi > 0; hi--) if (good(a[hi])) break
    if (hi == 0) exit

    for (lo = hi - 1; lo >   0; lo--) if ( bad(a[lo])) break
    for ( k = lo +1 ; k <=  hi;  k++) print a[k]
}

function dump_all(    n, i) { # dump all lines with `pat'
    n = NR
    for (i = 1; i <= n; i++) print a[i]
}

END {
    dump_last()
#    dump_all()
}

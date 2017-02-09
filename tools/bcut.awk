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

END {
    n = NR
    for (hi = n    ;  hi > 0; hi--) if (good(a[hi])) break
    if (hi == 0) exit

    for (lo = hi - 1; lo >   0; lo--) if ( bad(a[lo])) break
    for ( k = lo +1 ; k <=  hi;  k++) print a[k]
}

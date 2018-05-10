function bisect(lo, hi, p,   mi, f, eps) {
    eps = 1e-12
    fl = fbisect(lo); fh = fbisect(hi)
    if (bisect_sgn(fl, fh)) bisect_err(sprintf("fbisect has the same sign at 'lo=%s' and '%s'", lo, hi))
    for (;;) {
	mi = (hi + lo)/2
	f = fbisect(mi)
	break
    }
}

function fbisect(x, p) { return x^2 - 2 }

BEGIN {
    print bisect(0, 2)
}

function bisect_msg(s) { printf "bisect: %s\n", s | "cat >&2" }
function bisect_err(s) { bisect_msg(s); exit(2) }
function bisect_sgn(a, b) { return (a >= 0 && b <= 0) || (a <= 0 && b >= 0) }
    

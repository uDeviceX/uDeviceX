function bisect(lo, hi, p,   mi, fl, fh, fm, eps) {
    eps = 1e-12
    if (hi < lo)
	bisect_err(sprintf("hi='%s' < lo='%s'", lo, hi))
    fl = fbisect(lo, p); fh = fbisect(hi, p)
    if (!bisect_sgn(fl, fh))
	bisect_err(sprintf("fbisect has the same sign at '%s' and '%s'", lo, hi))
    for (;;) {
	mi = (hi + lo)/2
	if (hi - lo < eps) return mi
	fm = fbisect(mi, p)
	if (bisect_sgn(fl, fm)) {
	    hi = mi
	} else {
	    lo = mi; fl = fm
	}
    }
}
function bisect_msg(s) { printf "bisect: %s\n", s | "cat >&2" }
function bisect_err(s) { bisect_msg(s); exit(2) }
function bisect_sgn(a, b) { return (a >= 0 && b <= 0) || (a <= 0 && b >= 0) }
    

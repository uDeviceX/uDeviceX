function search(lo, hi, n, p,   fm, xm, i) {
    fm = 1e32; xm = lo
    for (i = 0; i < n; i++) {
	x = lo + i*(hi - lo)/(n - 1)
	fc = fsearch(x, p)
	if (search_less(fc, fm)) {
	    fm = fc; xm = x
	}
    }
    return xm
}
function search_msg(s) { printf "search: %s\n", s | "cat >&2" }
function search_err(s) { search_msg(s); exit(2) }
function search_abs(a) { return a > 0 ? a : -a }
function search_less(a, b) { return search_abs(a) < search_abs(b) }

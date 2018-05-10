function simp(lo, hi, n,   i, dx, A, B, C) {
    dx = (hi - lo)/n
    A = f(lo) + f(hi)
    for (i = 1; i <= n - 1; i += 2) {
	x = lo + i*dx
	B += f(x)
    }
    for (i = 2; i <= n - 2; i += 2) {
	x = lo + i*dx
	C += f(x)
    }
    return dx/3*(A + 4*B + 2*C)
}


function f(x) { return sin(x) }

BEGIN {
    print simp(0, 1, 10)
}

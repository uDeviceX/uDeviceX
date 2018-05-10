function simp(lo, hi, n, p,  i, dx, A, B, C) {
    dx = (hi - lo)/n
    A = f(lo, p) + f(hi, p)
    for (i = 1; i <= n - 1; i += 2) {
	x = lo + i*dx
	B += f(x, p)
    }
    for (i = 2; i <= n - 2; i += 2) {
	x = lo + i*dx
	C += f(x, p)
    }
    return dx/3*(A + 4*B + 2*C)
}

function f(x, p) { return sin(p*x) }

BEGIN {
    print simp(0, 1, 10, 1/2)
}

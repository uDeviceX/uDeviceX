function simp(lo, hi, n, p,  i, dx, A, B, C) {
    dx = (hi - lo)/n
    A = fsimp(lo, p) + fsimp(hi, p)
    for (i = 1; i <= n - 1; i += 2) {
	x = lo + i*dx
	B += fsimp(x, p)
    }
    for (i = 2; i <= n - 2; i += 2) {
	x = lo + i*dx
	C += fsimp(x, p)
    }
    return dx/3*(A + 4*B + 2*C)
}

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
function fsimp(x, p) { return sqrt(g(x, p)) }
function g(u, p) { return sin(u)^4*F1(cos(u), p)^2+cos(u)^2*sin(u)^2 }

function F1(q, p,   a, b, c) {
    a = p[1]; b = p[2]; c = p[3]
    return 5*c*q^4+3*b*q^2+a
}
function ini(   C0, C2, C4) {
    C0 = 0.2072
    C2 = 2.0026
    C4 = -1.12275
    a = (C4+C2+C0)/2
    c = C4/2
    b = -(2*C4+C2)/2
}
function volume(a, b, c,   pi) {
    pi = 3.141592653589793
    return (4*pi*(3*c/7+3*b/5+a))/3
}
function area(a, b, c,    lo, hi, n, pi, p) {
    pi = 3.141592653589793
    lo = 0; hi = pi; n = 200
    p[1] = a; p[2] = b; p[3] = c
    return 2*pi*simp(lo, hi, n, p)
}

BEGIN {
    ini()
    V = volume(a, b, c)
    A = area(a, b, c)
    print A, V
}

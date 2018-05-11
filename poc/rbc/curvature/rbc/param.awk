function fsimp(x, p) { return sqrt(gg(x, p)) }
function gg(u, p) { return sin(u)^4*F1(cos(u), p)^2+cos(u)^2*sin(u)^2 }
function F1(q, p,   a, b, c) {
    a = p[1]; b = p[2]; c = p[3]
    return 5*c*q^4+3*b*q^2+a
}
function F0(q, p,   a, b, c) {
    a = p[1]; b = p[2]; c = p[3]
    return c*q^5+b*q^3+a*q
}
function F2(q, p,   a, b, c) {
    a = p[1]; b = p[2]; c = p[3]
    return 20*c*q^3+6*b*q
}
function c1(u, p) {
    return (cos(u)*sin(u)^2*F2(cos(u), p)-F1(cos(u), p)) \
	/(sin(u)^2*F1(cos(u), p)^2+cos(u)^2)^(3/2)
}
function c2(u, p) {
    return -F1(cos(u), p)/sqrt(sin(u)^2*F1(cos(u), p)^2+cos(u)^2)
}
function z(u, p) { return F0(cos(u), p)}
function r(u) { return    sin(u) }
function volume(a, b, c,   pi) {
    pi = 3.141592653589793
    return 4*pi*(c/7+b/5+a/3)
}
function area(a, b, c,    lo, hi, n, pi, p) {
    pi = 3.141592653589793
    lo = 0; hi = pi; n = 200
    p[1] = a; p[2] = b; p[3] = c
    return 2*pi*simp(lo, hi, n, p)
}

function fbisect(a, p,   g, k, be, b, c, be0, A, V, pi) {
    pi = 3.141592653589793
    g = p[1]; k = p[2]; be = p[3]
    kga2bc(k, g, a)
    b = PAR_B; c = PAR_C
    A = area(a, b, c)
    V = volume(a, b, c)
    be0 = 6*sqrt(pi)*V/A^(3/2)
    return be - be0
}

function kga2bc(k, g, a) {
    PAR_B = (35*k)/6-(5*g)/2-(10*a)/3
    PAR_C = (-(35*k)/6)+(7*g)/2+(7*a)/3
}

function shape(p,   lo, hi, n, i, u, pi) {
    pi = 3.141592653589793
    lo = -pi; hi = pi; n = 200
    for (i = 0; i < n; i++) {
	u = lo + i*(hi - lo)/(n - 1)
	print r(u), z(u, p), c1(u, p), c2(u, p)
    }
}

BEGIN {
    # g = 0.1036; k = 0.375806; be = 0.644463
    g = 0.25; k = 0.375806; be = 0.644463
    p[1] = g; p[2] = k; p[3] = be
    lo = 0.0; hi = 10.0
    a = bisect(lo, hi, p)
    kga2bc(k, g, a)
    b = PAR_B; c = PAR_C
    bisect_msg("area:   " area(a, b, c))
    bisect_msg(sprintf("abc: %s %s %s", a, b, c))
    p[1] = a; p[2] = b; p[3] = c
    shape(p)
}

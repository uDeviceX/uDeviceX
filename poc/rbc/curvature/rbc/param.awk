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
    # g = 0.1036, k = 0.375806, be = 0.668099
    ini()
    pi = 3.141592653589793
    V = volume(a, b, c)
    A = area(a, b, c)
    print a + b + c         # g
    print a + 3/5*b + 3/7*c # k
    V0 = 3/4*pi
    print V/V0              # be
#    print A, V
}

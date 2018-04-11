#!/usr/bin/awk -f

function ini() {
    eps = 1e-23
    a0 = 0.0518; a1 = 2.0026; a2 = -4.491
    C5 = a2/16; C3 = -(a2+2*a1)/8; C1 = (a2+4*a1+16*a0)/16
}

function sgn(x) { return x > 0 ? 1 : -1 }
function ss(r) { return 2 * r } # sin('u)
function cc(r) { return sqrt(1 - ss(r)^2) }
function z(r)  { return f(cc(r)) }
function f(c)  { return    C5*c^5+  C3*c^3+C1*c }
function f1(c) { return  5*C5*c^4+3*C3*c^2+C1}
function f2(c) { return 20*C5*c^3+6*C3*c}
function L(r) { return (cc(r)*f2(cc(r))*ss(r)^3-f1(cc(r))*ss(r))/4 * sgn(r) }
function N(r) { return -(f1(cc(r))*ss(r)^3)/4 * sgn(r) }
function nn(r) { return sqrt(nx(r)^2 + nz(r)^2) + eps }
function eng(r,   n) {
    return ((L(r) + N(r))/2)^2/nn(r)^2
}
function nx(r) { return (f1(cc(r))*ss(r)^2)/2 * sgn(r)}
function nz(r) { return (cc(r)*ss(r))/4 * sgn(r)}

BEGIN {
    ini()
    lo =  -1/2
    hi =   1/2
    num = 300
    for (i = 0; i < num; i++) {
	r = lo + (hi - lo) * (i + 1) / num
	print r, z(r), nx(r), nz(r), L(r)/nn(r), N(r)/nn(r), eng(r)
    }
}

#
# plot [-5:5][-1:7] g u 1:2:($3/5):($4/5) with vector, "" u 1:2 w l lw 3

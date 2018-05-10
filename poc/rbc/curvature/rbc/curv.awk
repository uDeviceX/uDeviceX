#!/usr/bin/awk -f

function ini() {
    pi = 3.141592653589793
    a0 = 0.2072; a1 = 2.0026; a2 = -1.12275
    C1 = (a2+a1+a0)/2
    C3 = -(2*a2+a1)/2
    C5 = a2/2
}
function F0(c) { return C5*c^5+C3*c^3+C1*c   }
function F1(c) { return 5*C5*c^4+3*C3*c^2+C1 }
function F2(c) { return 20*C5*c^3+6*C3*c     }
function c1(u) {
    return (cos(u)*sin(u)^2*F2(cos(u))-F1(cos(u))) \
	/(sin(u)^2*F1(cos(u))^2+cos(u)^2)^(3/2)
}
function c2(u) {
    return -F1(cos(u))/sqrt(sin(u)^2*F1(cos(u))^2+cos(u)^2)
}
function z(u) { return F0(cos(u))}
function r(u) { return    sin(u) }
    
BEGIN {
    ini()
    lo = 0; hi = pi; n = 200
    for (i = 0; i < n; i++) {
	u = lo + i*(hi - lo)/(n - 1)
	print r(u), z(u), c1(u), c2(u)
    }
}

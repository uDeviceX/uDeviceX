#!/usr/bin/awk -f

function ini(   C0, C2, C4) {
    C0 = 0.2072
    C2 = 2.0026
    C4 = -1.12275
    #a = 0.54353;  b = 0.121435;  c = -0.561365
    a = 0.801527; b = -1.10455; c = 0.553028
}
function F0(q) { return c*q^5+b*q^3+a*q   }
function F1(q) { return 5*c*q^4+3*b*q^2+a }
function F2(q) { return 20*c*q^3+6*b*q    }
function F3(q) { return 60*c*q^2+6*b      }
function F4(q) { return 120*c*q           }

function c1(u) {
    return (cos(u)*sin(u)^2*F2(cos(u))-F1(cos(u))) \
	/(sin(u)^2*F1(cos(u))^2+cos(u)^2)^(3/2)
}
function c2(u) {
    return -F1(cos(u))/sqrt(sin(u)^2*F1(cos(u))^2+cos(u)^2)
}
function z(u) { return F0(cos(u))}
function r(u) { return    sin(u) }
function en(u) { return ((c1(u) + c2(u))/2)^2 }
function fo(u,   c10, c20, H, K) {
    c10 = c1(u); c20 = c2(u)
    H = (c10 + c20)/2
    K = c10 * c20
    return 2 * (2*H*(H^2 - K) + dsh(u))
}

function main(   lo, hi, n, i, u, pi) {
    pi = 3.141592653589793
    lo = 0; hi = pi; n = 200
    for (i = 0; i < n; i++) {
	u = lo + i*(hi - lo)/(n - 1)
	print r(u), z(u), en(u), fo(u)
    }
}
    
BEGIN {
    ini()
    main()
}

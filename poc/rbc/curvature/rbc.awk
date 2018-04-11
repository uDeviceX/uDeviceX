#!/usr/bin/awk -f

function ini() {
    eps = 1e-8
    pi = 3.141592653589793
    a0 = 0.0518; a1 = 2.0026; a2 = -4.491; D = 7.82
}

function sqrt0(x) { return x > eps ? sqrt(x) : sqrt(eps) }
function sgn(x) { return x > 0 ? 1 : -1 }
function abs(x) { return x > 0 ? x : -x }
function pow(a, b) {return a^b}
function f1(r){ return -(10*a2*pow(r,2)+(6*a1-2*a2)*r-a1+2*a0)/sqrt0(1-4*r) }
function f2(r){ return -(2*(6*r*(a2*(5*r-2)+a1)+a2-2*(a1+a0)))/(sqrt0(1-4*r)*(4*r-1 + eps)) }
function f(r) { return sqrt0(1-4*r)*(a2*r^2+a1*r+a0) }
function z(u) { return D * f(r(u))}
function r(u) { return u^2/D^2 }

function N(u) { return  \
	(sqrt(pow(D,2)-4*pow(u,2))*(20*a2*pow(u,7) \
                               +(12*pow(D,2)*a1-4*pow(D,2)*a2)*pow(u,5) \
                               +(4*pow(D,4)*a0-2*pow(D,4)*a1)*pow(u,3))) \
  /(4*pow(D,4)*pow(u,2)-pow(D,6)) }

function L(u) { return \
    -(320*a2*pow(u,7)+(96*pow(D,2)*a1-132*pow(D,2)*a2)*pow(u,5) \
                     +(12*pow(D,4)*a2-36*pow(D,4)*a1)*pow(u,3) \
                     +(2*pow(D,6)*a1-4*pow(D,6)*a0)*u) \
  /(sqrt(pow(D,2)-4*pow(u,2))*(4*pow(D,4)*pow(u,2)-pow(D,6)));
}

function nx(u){ return -(2*f1(r(u))*u^2)/D * sgn(u)}
function nz(u){ return abs(u)}
function nn(u){ return sqrt(nx(u)^2 + nz(u)^2) }

function eng(u) {
    return ( (L(u) + N(u))/(nn(u) + eps) )^2
}

BEGIN {
    ini()
    lo =  -D/2
    hi =   D/2
    num = 200
    for (i = 0; i < num; i++) {
	u = lo + (hi - lo) * (i + 1) / num
	phi = atan2(z(u), u); phi -= pi/2
	print u, z(u), nx(u), nz(u), L(u)/(nn(u) + eps), N(u)/(nn(u) + eps), eng(u)
    }
}

#
# plot [-5:5][-1:7] g u 1:2:($3/5):($4/5) with vector, "" u 1:2 w l lw 3

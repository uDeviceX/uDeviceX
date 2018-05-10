#!/usr/bin/awk -f

function ini() {
    a0 = 0.0518; a1 = 2.0026; a2 = -4.491
    C1 = (a2+4*a1+16*a0)/16
    C3 = ((-a2)-2*a1)/8    
    C5 = a2/16
}

function F0(c) { return C5*c^5+C3*c^3+C1*c   }
function F1(c) { return 5*C5*c^4+3*C3*c^2+C1 }
function F2(c) { return 20*C5*c^3+6*C3*c     }
    
function c1(u) {
    return (4*cos(u)*sin(u)^2*F2(cos(u))-4*F1(cos(u))) \
	/(4*sin(u)^2*F1(cos(u))^2+cos(u)^2)^(3/2)
}

function c2(u) {
    return -(4*F1(cos(u)))/sqrt(4*sin(u)^2*F1(cos(u))^2+cos(u)^2)
}

function z(u) { return F(cos(u))  }
function r(u) { return 1/2*sin(u) }
    

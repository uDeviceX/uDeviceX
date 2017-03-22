#include <cstdio>
#include "quadroots.h"

int main()
{
    //Real a = 1, b = -1, c = -1;
    Real a = 1., b = 200., c = -0.000015;

    RealComp t0 = 0, t1 = 0;
    
    if (robust_quadratic_roots(a, b, c, /**/ &t0, &t1))
    {
        printf("t0 = %.15g, f(t0) = %.15g\n", t0, (a * t0 + b) * t0 + c);
        printf("t1 = %.15g, f(t1) = %.15g\n", t1, (a * t1 + b) * t1 + c);
    }
    
    return 0;
}

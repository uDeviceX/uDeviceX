#include <cmath>
#include <utility>

#pragma once

typedef double RealComp;
typedef float Real;

bool robust_quadratic_roots(RealComp a, RealComp b, RealComp c, /**/ RealComp *t0, RealComp *t1)
{
    RealComp D;
    int sgnb;
    
    sgnb = b > 0 ? 1 : -1;
    D = b*b - 4*a*c;
    
    if (D < 0) return false;
    
    *t0 = (-b - sgnb * sqrt(D)) / (2 * a);
    *t1 = c / (a * (*t0));
    
    if (*t0 > *t1)
    std::swap(*t0, *t1);
        
    return true;
}

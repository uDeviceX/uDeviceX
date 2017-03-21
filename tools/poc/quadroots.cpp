#include <cstdio>
#include <cmath>
#include <utility>

//typedef double Real;
typedef float Real;

bool robust_quadratic_roots(Real a, Real b, Real c, /**/ Real *t0, Real *t1)
{
    Real D;
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

int main()
{
    //Real a = 1, b = -1, c = -1;
    Real a = 1., b = 200., c = -0.000015;


    Real t0 = 0, t1 = 0;
    
    if (robust_quadratic_roots(a, b, c, /**/ &t0, &t1))
    {
        printf("t0 = %.15g, f(t0) = %.15g\n", t0, (a * t0 + b) * t0 + c);
        printf("t1 = %.15g, f(t1) = %.15g\n", t1, (a * t1 + b) * t1 + c);
    }
    
    return 0;
}

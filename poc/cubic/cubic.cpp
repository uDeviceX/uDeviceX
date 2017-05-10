#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include "gsl_roots.h"

typedef float real;
//typedef double real;

const real lb[3] = {-1e-4, -1e-4, 0.};
const real ub[3] = {2e-4, 2e-4, 1.0};

// build polynomial knowing given roots h_i: product (x-h_i) = 0
void build_poly(const real h1, const real h2, const real h3, real *b, real *c, real *d)
{
    *b = -h1 - h2 - h3;
    *c = h1*h2 + h1*h3 + h2*h3;
    *d = -h1*h2*h3;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <N>\n", argv[0]);
        exit(1);
    }

    const int n = atoi(argv[1]);

    real maxe = -1;
    
    for (int i = 0; i < n; ++i)
    {
        real h1 = lb[0] + (ub[0]-lb[0]) * drand48();
        real h2 = lb[1] + (ub[1]-lb[1]) * drand48();
        real h3 = lb[2] + (ub[2]-lb[2]) * drand48();

        // reorder
        #define SWAP(a,b) do { auto tmp = b ; b = a ; a = tmp ; } while(0)
        if (h1 > h2) SWAP(h1, h2);
        if (h2 > h3) SWAP(h2, h3);
        if (h1 > h2) SWAP(h1, h2);
        #undef SWAP

        assert(h1 <= h2);
        assert(h2 <= h3);
        
        real b, c, d;
        build_poly(h1, h2, h3, &b, &c, &d);

        real x1, x2, x3;
        x1 = x2 = x3 = 0;
        gsl_roots::cubic(b, c, d, &x1, &x2, &x3);
        
        const real err1 = fabs(x1 - h1);
        const real err2 = fabs(x2 - h2);
        const real err3 = fabs(x3 - h3);

        const real err12 = err1 > err2 ? err1 : err2;
        const real e = err12 > err3 ? err12 : err3;

        //printf("error = %g\n", e);
        maxe = e > maxe ? e : maxe;
    }

    printf("max error = %g\n", maxe);
    
    return 0;
}

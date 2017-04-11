#include <cstdio>

enum {X, Y, Z};

typedef double real;

const real a0 = 0.0518, a1 = 2.0026, a2 = -4.491;
const real D0 = 7.82;


real frbc(const real x, const real y)
{
    const real rho = (x*x+y*y)/(D0*D0);
    const real s = 1 - 4 * rho;
    const real subg = (a0 + a1*rho + a2*rho*rho);
    
    return D0*D0 * s * subg * subg;
}

void dfrbc(const real x, const real y, real *gradxy)
{
    const real rho = (x*x+y*y)/(D0*D0);
    const real subg = a0 + a1 * rho + a2 * rho * rho;
    const real dgdrho = 2 * (1 - 4*rho) * (a1 + 2*a2*rho) * subg - 4 * subg * subg;
    
    gradxy[0] = 2 * x * dgdrho;
    gradxy[1] = 2 * y * dgdrho;
}

int main()
{

    real r0[3] = {4, 0, 0};
    real r1[3] = {3.5, 0, 0};
    real v0[3] = {r1[X] - r0[X], r1[Y] - r0[Y], r1[Z] - r0[Z]};
    
    real h = 0;
    real rh[3];
    
    auto newton_step = [&]() {

        rh[X] = r0[X] + h * v0[X];
        rh[Y] = r0[Y] + h * v0[Y];
        rh[Z] = r0[Z] + h * v0[Z];

        const real x = rh[X], y = rh[Y], z = rh[Z];

        const real f = frbc(x, y) - z*z;
        
        real gradxy[2];
        dfrbc(x, y, gradxy);

        const real dxdh = v0[X];
        const real dydh = v0[Y];
        const real dzdh = v0[Z];
                
        const real df = dxdh * gradxy[X] + dydh * gradxy[Y] - 2 * dzdh * z;

        printf("f = %f, df = %.6e\n", f, df);
        
        h = h - f / df;
    };

    for (int step = 0; step < 10; ++step)
    {
        newton_step();
        printf("h = %f\n", h);
    }

    printf("rw = %f %f %f\n", rh[X], rh[Y], rh[Z]);
    
    return 0;
}

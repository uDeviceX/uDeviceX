#include <cstdio>
#include <cstdlib>
#include <cmath>

struct Particle {float r[3], v[3];};
enum {X, Y, Z};

#include "roots.h"
#include "bb.h"

enum {XD = 1, YD = 1, ZD = 1};

void gen_point(float *r)
{
    r[X] = drand48() * XD;
    r[Y] = drand48() * YD;
    r[Z] = drand48() * ZD;
}

void gen_points(const int n, float *rr)
{
    for (int i = 0; i < n; ++i)
    gen_point(rr + 3*i);
}

void vel(const int n, const float *rr0, const float *rr1, const float dt, float *vv)
{
    for (int i = 0; i < 3*n; ++i)
    vv[i] = (rr1[i] - rr0[i]) / dt;
}

void dump_t(const float *A, const float *B, const float *C, const char *fname)
{
    
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <nsamples>\n", argv[0]);
        exit(1);
    }

    const float dt = 0.1f;

    const int n = atoi(argv[1]);
    
    float *AA0, *BB0, *CC0, *rr0;
    float *AA1, *BB1, *CC1, *rr1;
    float *AAV, *BBV, *CCV, *rrV;

#define alloc(VV) VV = new float[3*n];
    alloc(AA0); alloc(BB0); alloc(CC0); alloc(rr0);
    alloc(AA1); alloc(BB1); alloc(CC1); alloc(rr1);
    alloc(AAV); alloc(BBV); alloc(CCV); alloc(rrV);

    gen_points(n, AA0); gen_points(n, BB0); gen_points(n, CC0); gen_points(n, rr0);
    gen_points(n, AA1); gen_points(n, BB1); gen_points(n, CC1); gen_points(n, rr1);

    vel(n, AA0, AA1, dt, AAV);
    vel(n, BB0, BB1, dt, BBV);
    vel(n, CC0, CC1, dt, CCV);
    vel(n, rr0, rr1, dt, rrV);

    for (int i = 0; i < n; ++i)
    {
        float h, rw[3];
        Particle p0;

        p0.r[X] = rr0[3*i+X]; p0.r[Y] = rr0[3*i+Y]; p0.r[Z] = rr0[3*i+Z];
        p0.v[X] = rrV[3*i+X]; p0.v[Y] = rrV[3*i+Y]; p0.v[Z] = rrV[3*i+Z];
        
        const BBState bbstate = intersect_triangle(AA0 + 3*i, BB0 + 3*i, CC0 + 3*i,
                                                   AAV + 3*i, BBV + 3*i, CCV + 3*i,
                                                   &p0, dt, /**/ &h, rw);
        if (bbstate == BB_SUCCESS)
        {
            
        }
    }
    
    delete[] AA0; delete[] BB0; delete[] CC0; delete[] rr0;
    delete[] AA1; delete[] BB1; delete[] CC1; delete[] rr1;
    delete[] AAV; delete[] BBV; delete[] CCV; delete[] rrV;
    
    return 0;
}

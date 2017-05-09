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

void dump_l(const float *A, const float *B, const char *fname)
{
    FILE *f = fopen(fname, "w");
#define pr(...) fprintf (f, __VA_ARGS__)
    pr("path\n");
    pr("%g, %g, %g\n", A[X], A[Y], A[Z]);
    pr("%g, %g, %g\n", B[X], B[Y], B[Z]);
#undef pr
  fclose(f);
}
    
void dump_t(const float *A, const float *B, const float *C, const char *fname)
{
    FILE *f = fopen(fname, "w");
#define pr(...) fprintf (f, __VA_ARGS__)
    pr("# vtk DataFile Version 2.0\n");
    pr("generated with bbgen\n");
    pr("ASCII\n");
    pr("DATASET POLYDATA\n");
    pr("POINTS 3 float\n");
    pr("%g %g %g\n", A[X], A[Y], A[Z]);
    pr("%g %g %g\n", B[X], B[Y], B[Z]);
    pr("%g %g %g\n", C[X], C[Y], C[Z]);
    pr("POLYGONS 1 4\n");
    pr("3 0 1 2\n");
#undef pr
  fclose(f);
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
#undef alloc
    
    gen_points(n, AA0); gen_points(n, BB0); gen_points(n, CC0); gen_points(n, rr0);
    gen_points(n, AA1); gen_points(n, BB1); gen_points(n, CC1); gen_points(n, rr1);

    vel(n, AA0, AA1, dt, AAV);
    vel(n, BB0, BB1, dt, BBV);
    vel(n, CC0, CC1, dt, CCV);
    vel(n, rr0, rr1, dt, rrV);

    int idf = 0;

    int states[4] = {0};
    
    for (int i = 0; i < n; ++i)
    {
        float h, rw[3];
        Particle p0;

        const float *A0 = AA0 + 3*i;
        const float *B0 = BB0 + 3*i;
        const float *C0 = CC0 + 3*i;

        const float *AV = AAV + 3*i;
        const float *BV = BBV + 3*i;
        const float *CV = CCV + 3*i;
        
        p0.r[X] = rr0[3*i+X]; p0.r[Y] = rr0[3*i+Y]; p0.r[Z] = rr0[3*i+Z];
        p0.v[X] = rrV[3*i+X]; p0.v[Y] = rrV[3*i+Y]; p0.v[Z] = rrV[3*i+Z];
        
        const BBState bbstate = intersect_triangle(A0, B0, C0, AV, BV, CV,
                                                   &p0, dt, /**/ &h, rw);

        ++ states[bbstate];
        
        if (bbstate == BB_SUCCESS)
        {
            const float Aw[3] = {A0[X] + h * AV[X], A0[Y] + h * AV[Y], A0[Z] + h * AV[Z]};
            const float Bw[3] = {B0[X] + h * BV[X], B0[Y] + h * BV[Y], B0[Z] + h * BV[Z]};
            const float Cw[3] = {C0[X] + h * CV[X], C0[Y] + h * CV[Y], C0[Z] + h * CV[Z]};
                            
            char ft0[256] = {0}; sprintf(ft0, "data/t0-%.05d.vtk", idf);
            char ftw[256] = {0}; sprintf(ftw, "data/tw-%.05d.vtk", idf);
            char ft1[256] = {0}; sprintf(ft1, "data/t1-%.05d.vtk", idf);
            char fl0[256] = {0}; sprintf(fl0, "data/l0-%.05d.lines", idf);
            char fl1[256] = {0}; sprintf(fl1, "data/l1-%.05d.lines", idf);

            dump_t(A0, B0, C0, ft0);
            dump_t(Aw, Bw, Cw, ftw);
            dump_t(AA1 + 3*i, BB1 + 3*i, CC1 + 3*i, ft1);

            dump_l(rr0 + 3*i, rw, fl0);
            dump_l(rw, rr1 + 3*i, fl1);

            ++idf;
        }
        
    }

    printf("%d success, %d nocross, %d wrong triangle, %d hfailed\n",
           states[0], states[1], states[2], states[3]);
    
    delete[] AA0; delete[] BB0; delete[] CC0; delete[] rr0;
    delete[] AA1; delete[] BB1; delete[] CC1; delete[] rr1;
    delete[] AAV; delete[] BBV; delete[] CCV; delete[] rrV;
    
    return 0;
}

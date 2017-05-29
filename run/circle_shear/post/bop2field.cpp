#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "reader.h"


void v_r(const float *com, const float Rmin, const float Rmax, const int nR,
              const float *data, const long n, const int nvars, float *av, int *counts)
{
    const int nf = nvars - 3;
    if (nf > 0) memset(av, 0, nR * nf * sizeof(float));
    memset(counts, 0, nR*sizeof(int));

    const float dr = (Rmax - Rmin) / nR;
    
    for (long i = 0; i < n; ++i)
    {
        const long base = i*nvars;
        const float *ro = data + base;
        const float *fo  = data + base + 3;

        const float r[3] = {ro[0] - com[0], ro[1] - com[1], ro[2] - com[2]};
        const float r_ = sqrt(r[0]*r[0] + r[1]*r[1]); // suppose cylinder in z direction

        const int ir = (int) floor( (r_-Rmin) / dr );

        if (ir < 0 || ir >= nR) continue;

        ++counts[ir];

        // transform velocity to polar coords

        float f[nf];
        memcpy(f, fo, nf*sizeof(float));
        
        if (nf >= 3)
        {
            const float sc = 1.f / r_;
            
            f[0] = sc * (r[0] * fo[0] + r[1] * fo[1]);
            f[1] = sc * (r[1] * fo[0] - r[0] * fo[1]);
        }
        
        for (int j = 0; j < nf; ++j)
        av[ir * nf + j] += f[j];
    }

    for (int i = 0; i < nR; ++i)
    for (int j = 0; j < nf; ++j)
    av[i * nf + j] /= counts[i] > 0 ? counts[i] : 1;
}

void v_th(const float *com, const float Rmin, const float Rmax, const int nth,
          const float *data, const long n, const int nvars, float *av, int *counts)
{
    const int nf = nvars - 3;
    if (nf > 0) memset(av, 0, nth * nf * sizeof(float));
    memset(counts, 0, nth*sizeof(int));

    const float dth = 2*M_PI / nth;
    
    for (long i = 0; i < n; ++i)
    {
        const long base = i*nvars;
        const float *ro = data + base;
        const float *fo  = data + base + 3;

        const float r[3] = {ro[0] - com[0], ro[1] - com[1], ro[2] - com[2]};
        const float r_ = sqrt(r[0]*r[0] + r[1]*r[1]); // suppose cylinder in z direction

        if (r_ < Rmin || r_ >= Rmax) continue;

        const float thetap = atan2(r[1], r[0]);
        const float theta = thetap < 0 ? 2 * M_PI + thetap : thetap;

        const int it = int (theta / dth);
        
        ++counts[it];

        // transform velocity to radial coords

        float f[nf];
        memcpy(f, fo, nf*sizeof(float));
        
        if (nf >= 3)
        {
            const float sc = 1.f / r_;
            
            f[0] = sc * (r[0] * fo[0] + r[1] * fo[1]);
            f[1] = sc * (r[1] * fo[0] - r[0] * fo[1]);
        }
        
        for (int j = 0; j < nf; ++j)
        av[it * nf + j] += f[j];
    }

    for (int i = 0; i < nth; ++i)
    for (int j = 0; j < nf; ++j)
    av[i * nf + j] /= counts[i] > 0 ? counts[i] : 1;
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <in1.bop> <in2.bop> ...\n", argv[0]);
        exit(1);
    }

    ReadData d;
    init(&d);
    {
        const int nd = argc-1;
        ReadData *dd = new ReadData[nd];

        for (int i = 0; i < nd; ++i)
        {
            init(dd + i);
            read(argv[1+i], dd + i);
        }

        concatenate(nd, dd, /**/ &d);
        for (int i = 0; i < nd; ++i) finalize(dd + i);
        delete[] dd;
    }
    summary(&d);

    const int nr = 50;
    const int nt = 50;
    
    const float rmin = 5.f;
    const float rmax = 6.f;
    const float drt = 0.01f;
    const float com[3] = {32, 32, 4};

    const int nf = d.nvars - 3;
    
    int *countsr = new int[nr];
    int *countst = new int[nt];
    
    float *avr = new float[nr * nf];
    float *avt = new float[nt * nf];
    
    v_r(com, rmin, rmax,      nr, d.fdata, d.n, d.nvars, /**/ avr, countsr);
    v_th(com, rmin, rmin+drt, nt, d.fdata, d.n, d.nvars, /**/ avt, countst);
        
    FILE *fr = fopen("f_r.txt", "w");
    FILE *ft = fopen("f_t.txt", "w");

    const float dr = (rmax - rmin) / nr;
    const float dt = 2 * M_PI / nt;

    for (int i = 0; i < nr; ++i)
    {
        fprintf(fr, "%.6e %d", rmin + i*dr, countsr[i]);
        for (int j = 0; j < nf; ++j)
        fprintf(fr, " %.6e", avr[i * nf + j]);
        fprintf(fr, "\n");
    }

    for (int i = 0; i < nt; ++i)
    {
        fprintf(ft, "%.6e %d", i*dt, countst[i]);
        for (int j = 0; j < nf; ++j)
        fprintf(ft, " %.6e", avt[i * nf + j]);
        fprintf(ft, "\n");
    }

    fclose(fr);
    fclose(ft);

    delete[] countsr;
    delete[] countst;

    delete[] avr;
    delete[] avt;

    finalize(&d);
    
    return 0;
}

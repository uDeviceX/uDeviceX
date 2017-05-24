#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "reader.h"


void radial_v(const float *com, const float Rmin, const float Rmax, const int nR,
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

        // transform velocity to radial coords

        float f[nf];
        memcpy(f, fo, nf*sizeof(float));
        
        if (nf >= 3)
        {
            const float sc = 1.f / r_;
            
            f[0] = sc * (r[0] * fo[0] + r[1] * fo[1]);
            f[1] = sc * (r[1] * fo[1] - r[0] * fo[0]);
        }
        
        for (int j = 0; j < nf; ++j)
        av[ir * nf + j] += f[j];
    }

    for (int i = 0; i < nR; ++i)
    for (int j = 0; j < nf; ++j)
    av[i * nf + j] /= counts[i] > 0 ? counts[i] : 1;
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <out.txt> <in1.bop> <in2.bop> ...\n", argv[0]);
        exit(1);
    }

    ReadData d;
    init(&d);
    {
        const int nd = argc-2;
        ReadData *dd = new ReadData[nd];

        for (int i = 0; i < nd; ++i)
        {
            init(dd + i);
            read(argv[2+i], dd + i);
        }

        concatenate(nd, dd, /**/ &d);
        for (int i = 0; i < nd; ++i) finalize(dd + i);
        delete[] dd;
    }
    summary(&d);

    const int nr = 50;
    const float rmin = 5.f;
    const float rmax = 6.f;
    const float com[3] = {32, 32, 4};

    const int nf = d.nvars - 3;
    
    int *counts = new int[nr];
    float *av = new float[nr * nf];
    
    radial_v(com, rmin, rmax, nr, d.fdata, d.n, d.nvars, /**/ av, counts);
        
    FILE *f = fopen(argv[1], "w");

    const float dr = (rmax - rmin) / nr;

    for (int i = 0; i < nr; ++i)
    {
        fprintf(f, "%.6e %d", rmin + i*dr, counts[i]);
        for (int j = 0; j < nf; ++j)
        fprintf(f, " %.6e", av[i * nf + j]);
        fprintf(f, "\n");
    }

    fclose(f);

    finalize(&d);
    
    return 0;
}

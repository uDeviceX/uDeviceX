#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "reader.h"

#define ERR(...) do {                            \
        fprintf(stderr,__VA_ARGS__);             \
        exit(1);                                 \
    } while (0);

int separator(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
    if (strcmp("--", argv[i]) == 0) return i;
    return -1;
}

void read_data(const char *fpp, ReadData *dpp, const char *fii, ReadData *dii) {
    read(fpp, dpp);
    read(fii, dii);

    if (dpp->type != FLOAT) ERR("expected float data form <%s>\n", fpp);
    if (dii->type != INT)   ERR("expected int   data form <%s>\n", fii);
}

int max_index(const int *ii, const int n) {
    int m = -1;
    for (int i = 0; i < n; ++i) m = m < ii[i] ? ii[i] : m;
    return m;
}

void pp2rr_sorted(const int *ii, const float *fdata, const int n, const int stride, /**/ float *rr) {
    for (int j = 0; j < n; ++j) {
        const int i = ii[j];
        const float *r = fdata + j * stride;
        for (int c = 0; c < 3; ++c)
        rr[3*i + c] = r[c];
    }
}

enum {XS=16, YS=32, ZS=16};

void glbrr(const float *rrp, const int n, /**/ float *rrc, float *ddL) {
    const float dL[3] = {XS, YS, ZS};
    
    for (int i = 0; i < 3*n; ++i) {
        const int d = i%3;
        const float dl = rrp[i] - rrc[i];
        const float sign = dl > 0 ? 1.f : -1.f;
        if (fabs(dl) > dL[d]/2) {
            ddL[i] += sign * dL[d];
            rrc[i] += sign * dL[d];
        }
    }
}

float MSD(const float *rr0, const float *rr, const int buffsize, const int n) {
    float sumsq = 0.f;

    for (int i = 0; i < buffsize; ++i) {
        const float *r0 = rr0 + 3*i;
        const float *r  = rr  + 3*i;
        for (int c = 0; c < 3; ++c) {
            const float dr = r[c] - r0[c];
            sumsq += dr*dr;
        }
    }
    
    return sumsq / n;
}

int main(int argc, char **argv) {

    if (argc < 4) {
        fprintf(stderr, "Usage: po.msd <inpp-*.bop> -- <inii-*.bop>\n");
        exit(1);
    }

    const int sep = separator(argc, argv);
    const int nin = sep - 1;

    if (nin < 2) ERR("Need more than one file\n");
    
    char **ffpp = argv + 1;
    char **ffii = ffpp + sep;
    
    ReadData dpp0, dii0, dpp, dii;
    init(&dpp0); init(&dii0);
    
    read_data(ffpp[0], &dpp0, ffii[0], &dii0);

    const int buffsize = max_index(dii0.idata, dii0.n);

    float *rr0 = new float[3*buffsize]; /* initial  positions     */
    float *rrc = new float[3*buffsize]; /* current  positions     */
    float *rrp = new float[3*buffsize]; /* previous positions     */
    float *ddL = new float[3*buffsize]; /* helper for periodic BC */

    memset(rr0, 0, 3*buffsize*sizeof(float));
    memset(ddL, 0, 3*buffsize*sizeof(float));
    pp2rr_sorted(dii0.idata, dpp0.fdata, dpp0.n, dpp0.nvars, /**/ rr0);
    memcpy(rrp, rr0, 3*buffsize*sizeof(float));
    
    for (int i = 1; i < nin; ++i) {
        init(&dpp);  init(&dii);
        read_data(ffpp[i], &dpp, ffii[i], &dii);

        memset(rrc, 0, 3*buffsize*sizeof(float));
        pp2rr_sorted(dii.idata, dpp.fdata, dpp.n, dpp.nvars, /**/ rrc);
        glbrr(rrp, buffsize, /**/ rrc, ddL);

        const float msd = MSD(rr0, rrc, buffsize, dpp.n);

        printf("%f\n", msd);
        
        finalize(&dpp);  finalize(&dii);
        memcpy(rrp, rrc, 3*buffsize*sizeof(float));
    }

    delete[] rr0; delete[] rrc;    
    finalize(&dpp0); finalize(&dii0);
    return 0;
}

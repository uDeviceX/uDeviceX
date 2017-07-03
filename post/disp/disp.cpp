#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "reader.h"

int X, Y, Z;

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

void updddL(const float *rrp, const float *rrc, const int n, /**/ float *ddL) {
    const int dL[3] = {X, Y, Z};
    
    for (int i = 0; i < 3*n; ++i) {
        const int d = i%3;
        const float dl = rrc[i] - rrp[i];
        const float sign = dl > 0 ? 1.f : -1.f;
        if (fabs(dl) > dL[d]/2) {
            ddL[i] -= sign * dL[d];
        }
    }
}

void disp0(const float *rp, const float *rc, float *dr) {
    const int dL[3] = {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        dr[c] = rc[c] - rp[c];
        const float sign = dr[c] > 0 ? 1.f : -1.f;
        dr[c] += fabs(dr[c]) > dL[c]/2 ? sign * dL[c] : 0.f;        
    }
}

void displ(const float *rrp, const float *rrc, const int buffsize, /**/ float *ddr) {
    for (int i = 0; i < buffsize; ++i) {
        const float *rp = rrp + 3*i;
        const float *rc = rrc + 3*i;
        float *dr       = ddr + 3*i;
        disp0(rp, rc, /**/ dr);
    }
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: po.disp <X> <Y> <Z> <rr-*.bop> -- <ii-*.bop>\n");
        exit(1);
    }
    int iarg = 1;
    X = atoi(argv[iarg++]);
    Y = atoi(argv[iarg++]);
    Z = atoi(argv[iarg++]);

    const int sep = separator(argc, argv);
    const int nin = sep - iarg;

    
    
    return 0;
}

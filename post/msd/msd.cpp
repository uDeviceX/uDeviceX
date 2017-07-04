#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "bop_reader.h"

#include "macros.h"
#include "pp_id.h"

int X, Y, Z;

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

float MSD(const float *rr0, const float *rr, const float *ddL, const int buffsize, const int n) {
    float sumsq = 0.f;

    for (int i = 0; i < buffsize; ++i) {
        const float *r0 = rr0 + 3*i;
        const float  *r = rr  + 3*i;
        const float *dL = ddL + 3*i;
        for (int c = 0; c < 3; ++c) {
            const float dr = r[c] + dL[c] - r0[c];
            sumsq += dr*dr;
        }
    }
    
    return sumsq / n;
}

int main(int argc, char **argv) {

    if (argc < 7) {
        fprintf(stderr, "Usage: po.msd <X> <Y> <Z> <inpp-*.bop> -- <inii-*.bop>\n");
        exit(1);
    }
    int iarg = 1;
    X = atoi(argv[iarg++]);
    Y = atoi(argv[iarg++]);
    Z = atoi(argv[iarg++]);

    const int sep = separator(argc, argv);
    const int nin = sep - iarg;

    if (nin < 2) ERR("Need more than one file\n");
    
    char **ffpp = argv + iarg;
    char **ffii = ffpp + nin + 1;
    
    BopData dpp0, dii0, dpp, dii;
    init(&dpp0); init(&dii0);
    
    read_data(ffpp[0], &dpp0, ffii[0], &dii0);

    const int buffsize = max_index(dii0.idata, dii0.n) + 1;

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
        // printf("%s -- %s\n", ffpp[i], ffii[i]);
        
        read_data(ffpp[i], &dpp, ffii[i], &dii);

        memset(rrc, 0, 3*buffsize*sizeof(float));
        pp2rr_sorted(dii.idata, dpp.fdata, dpp.n, dpp.nvars, /**/ rrc);
        updddL(rrp, rrc, buffsize, /**/ ddL);

        const float msd = MSD(rr0, rrc, ddL, buffsize, dpp.n);

        printf("%f\n", msd);
        
        finalize(&dpp);  finalize(&dii);
        memcpy(rrp, rrc, 3*buffsize*sizeof(float));
    }

    delete[] rr0; delete[] rrc;
    delete[] rrp; delete[] ddL;
    finalize(&dpp0); finalize(&dii0);
    return 0;
}

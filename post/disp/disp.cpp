#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "bop_reader.h"
#include "bop_writer.h"

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

void read_data(const char *fpp, BopData *dpp, const char *fii, BopData *dii) {
    read(fpp, dpp);
    read(fii, dii);

    if (dpp->type != FLOAT && dpp->type != FASCII) ERR("expected float data form <%s>\n", fpp);
    if (dii->type != INT   && dii->type != IASCII) ERR("expected int data form <%s>\n", fii);
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

enum {EMPTY=0, OCCUPIED=1};
void empty_tags(const int bufsize, int *tags) {
    for (int i = 0; i < bufsize; ++i) tags[i] = EMPTY;
}

void compute_tags(const int *ii, const int n, int *tags) {
    for (int j = 0; j < n; ++j) {
        const int i = ii[j];
        tags[i] = OCCUPIED;
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

void disp(const float *rrp, const float *rrc, const int buffsize, /**/ float *ddr) {
    for (int i = 0; i < buffsize; ++i) {
        const float *rp = rrp + 3*i;
        const float *rc = rrc + 3*i;
        float *dr       = ddr + 3*i;
        disp0(rp, rc, /**/ dr);
    }
}

void outname(const char *inrr, char *out) {
    const int l = strlen(inrr);
    memcpy(out, inrr, l * sizeof(char));
    const int strt = l - 4;
    const char newext[] = ".disp.bop";
    memcpy(out + strt, newext, sizeof(newext));
}

void dump(const char *fout, const float *rr, const float *ddr, const int *tags, const int buffsize, /*w*/ float *work) {
    /* fill work buffer */
    int j = 0;
    for (int i = 0; i < buffsize; ++i)
    if (tags[i] == OCCUPIED) {
        for (int c = 0; c < 3; ++c) {
            work[6 * j + 0 + c] = rr[3 * i + c];
            work[6 * j + 3 + c] = ddr[3 * i + c];
        }
        ++j;
    }

    /* dump */
    BopData d;
    init(&d);
    d.n = j;
    d.nvars = 6;
    d.type = FLOAT;
    //d.type = FASCII;
    d.vars = new Cbuf[d.nvars];
    strncpy(d.vars[0].c, "x", 4);
    strncpy(d.vars[1].c, "y", 4);
    strncpy(d.vars[2].c, "z", 4);
    strncpy(d.vars[3].c, "dx", 4);
    strncpy(d.vars[4].c, "dy", 4);
    strncpy(d.vars[5].c, "dz", 4);

    d.fdata = new float[d.n * d.nvars];
    memcpy(d.fdata, work, d.n * d.nvars * sizeof(float));

    write(fout, d);
    
    finalize(&d);
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

    if (nin < 2) ERR("Need more than one file\n");
    
    char **ffpp = argv + iarg;
    char **ffii = ffpp + nin + 1;
    
    BopData dpp, dii;

    init(&dpp); init(&dii);
    read_data(ffpp[0], &dpp, ffii[0], &dii);

    const int buffsize = max_index(dii.idata, dii.n) + 1;

    float *rrc = new float[3*buffsize]; /* current  positions     */
    float *rrp = new float[3*buffsize]; /* previous positions     */
    float *ddr = new float[3*buffsize]; /* displacements          */
    float *rrw = new float[6*buffsize]; /* work                   */
    int  *tags = new int[buffsize];     /* tags: particle with this id or not? */

    memset(rrp, 0, 3*buffsize*sizeof(float));
    pp2rr_sorted(dii.idata, dpp.fdata, dpp.n, dpp.nvars, /**/ rrp);

    empty_tags(buffsize, /**/ tags);
    compute_tags(dii.idata, dii.n, /**/ tags);

    finalize(&dpp); finalize(&dii);
    
    for (int i = 0; i < nin-1; ++i) {
        init(&dpp);  init(&dii);
        char fout[1024] = {0};
        outname(ffpp[i], /**/ fout);
        printf("%s -- %s -> %s\n", ffpp[i], ffii[i], fout);

        read_data(ffpp[i+1], &dpp, ffii[i+1], &dii);
        pp2rr_sorted(dii.idata, dpp.fdata, dpp.n, dpp.nvars, /**/ rrc);

        disp(rrp, rrc, buffsize, /**/ ddr);

        dump(fout, rrp, ddr, tags, buffsize, /*w*/ rrw);
        compute_tags(dii.idata, dii.n, /**/ tags);
        
        finalize(&dpp);  finalize(&dii);

        {   /* swap */
            float *const tmp = rrp;
            rrp = rrc; rrc = tmp;
        }
    }

    delete[] rrp; delete[] rrc;
    delete[] ddr; delete[] rrw;
    delete[] tags;
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "bop_common.h"
#include "bop_serial.h"

#include "macros.h"
#include "pp_id.h"

void *emalloc(size_t size) {
    void *b;
    b = malloc(size);
    if (b == NULL) {
        fprintf(stderr, "u.paste: malloc fails: sz = %ld\n", size);
        exit(2);
    }
    return b;
}

void inc0(void **pp, int sz) { /* advance by number of bytes */
    char *p;
    p = (char*)*pp;
    p += sz;
    *pp = p;
}
void *inc(void **p, int sz) {
    void *p0;
    p0 = *p;
    inc0(p, sz);
    return p0;
}

void buffer0(int n,   float *f, int nf,   int *i, int ni, /**/ long *pnw, void *b) {
    int sz, nw;
    nw = 0;
    while (n--) {
        sz = nf*sizeof(float);
        memcpy(b, f, sz);
        inc(&b, sz); nw += sz; f += nf;

        sz = ni*sizeof(int);
        memcpy(b, i, sz);
        inc(&b, sz); nw += sz; i += ni;
    }
    *pnw = nw;
}

void buffer(BopData *f, BopData *i, /**/ long *pnw, void *b) {
    long n;
    int nf, ni;
    bop_get_n(f, &n);
    bop_get_nvars(f, &nf);
    bop_get_nvars(i, &ni);
    buffer0(n,
            (float*)bop_get_data(f), nf,
            (int*  )bop_get_data(i), ni, /**/ pnw, b);
}


void main1(BopData *f, BopData *i, void *b) {
    long nw;
    buffer(f, i, /**/ &nw, b);
    //fprintf(stderr, "nw = %ld\n", nw);
    fwrite(b, 1, nw, stdout);
}

void main2(BopData *f, BopData *i) {
    long n, sz;
    int ni, nf;
    void *b; /* buffer */
    bop_get_n(f, &n);;
    bop_get_nvars(f, &nf);
    bop_get_nvars(i, &ni);
    sz = n * (nf*sizeof(float) + ni*sizeof(int));
    b = emalloc(sz);
    main1(f, i, b);
    free(b);
}

void main3(BopData *f, BopData *i) {
    long nf, ni;
    BopType tf, ti;
    int nvi, nvf;

    bop_get_n(f, &nf);
    bop_get_n(i, &ni);
    bop_get_nvars(f, &nvf);
    bop_get_nvars(i, &nvi);
    bop_get_type(f, &tf);
    bop_get_type(i, &ti);
    
    assert(nf  == ni);
    assert(nvf == 6); /* x, y, z, vx, vy, vz */
    assert(tf  == BopFLOAT);

    assert(nvi == 1); /* color */
    assert(ti  == BopINT);
    main2(f, i);
}

void main4(const char *f0, const char *i0) {
    BopData *f, *i;
    bop_ini(&f);
    bop_ini(&i);
    read_data(f0, f, i0, i);
    main3(f, i);
    bop_fin(f);
    bop_fin(i);
}

int main(int argc, char **argv) {
    const char *f, *i; /* [f]loat, [i]neger */
    f = argv[1];
    i = argv[2];
    //fprintf(stderr, "<%s> <%s>\n", f, i);
    main4(f, i);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "bop_common.h"
#include "bop_reader.h"

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

void main2(BopData *f, BopData *i, void *b) {
}

void main2(BopData *f, BopData *i) {
    long n, sz;
    int ni, nf;
    void *b; /* buffer */
    n = f->n;
    nf = f->nvars;
    ni = i->nvars;
    sz = n * (nf*sizeof(float) + ni*sizeof(int));
    b = emalloc(sz);
    main2(f, i, b);
    free(b);
}

void main3(BopData *f, BopData *i) {
    assert(f->n     == i->n);
    assert(f->nvars == 6); /* x, y, z, vx, vy, vz */
    assert(f->type == FLOAT);
    
    assert(i->nvars == 1); /* color */
    assert(i->type == INT);
    summary(f);
    summary(i);
    main2(f, i);
}

void main4(const char *f0, const char *i0) {
    BopData f, i;
    init(&f); init(&i);
    read_data(f0, &f, i0, &i);
    main3(&f, &i);
    finalize(&f);  finalize(&i);
}

int main(int argc, char **argv) {
    const char *f, *i; /* [f]loat, [i]neger */
    f = argv[1];
    i = argv[2];
    fprintf(stderr, "<%s> <%s>\n", f, i);
    main4(f, i);
}

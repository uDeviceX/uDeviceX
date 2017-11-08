#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "bop_common.h"
#include "bop_reader.h"

#include "macros.h"
#include "pp_id.h"

void main0(BopData *f, BopData *i) {
    assert(f->n == i->n);
    summary(f);
    summary(i);
}

void main1(const char *f0, const char *i0) {
    BopData f, i;
    init(&f); init(&i);
    read_data(f0, &f, i0, &i);
    main0(&f, &i);
    finalize(&f);  finalize(&i);
}

int main(int argc, char **argv) {
    const char *f, *i; /* [f]loat, [i]neger */
    f = argv[1];
    i = argv[2];
    fprintf(stderr, "<%s> <%s>\n", f, i);
    main1(f, i);
}

#include <stdio.h>

#include "inc/type.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"

void punto_dump(long n, Particle *pp, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    float *r, *v;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        rcode = fprintf(f, "%16.10e %16.10e %16.10e %16.10e %16.10e %16.10e\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

#include <stdio.h>

#include "inc/type.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"

#define VFRMT "%16.10e %16.10e %16.10e"

void punto_write_pp(long n, const Particle *pp, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        rcode = fprintf(f, VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

void punto_write_pp_ff(long n, const Particle *pp, const Force *ff, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v, *a;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        a = ff[i].f;
        rcode = fprintf(f, VFRMT " " VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z], a[X], a[Y], a[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

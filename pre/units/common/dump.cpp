#include <stdio.h>

#include "type.h"
#include "dump.h"

enum {X, Y, Z};

void write_pp(int n, const Particle *pp, FILE *stream) {
    int i;
    Particle p;
    for (i = 0; i < n; ++i) {
        p = pp[i];
        fprintf(stream, "%f %f %f %f %f %f\n", p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);
    }   
}

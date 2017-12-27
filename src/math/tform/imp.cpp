#include <stdio.h>

#include "msg.h"
#include "utils/imp.h"
#include "utils/error.h"

struct Tform { float o[3], s[3]; };
void tform_ini(Tform **pq) {
    Tform *q;
    UC(emalloc(sizeof(Tform), (void**)&q));
    *pq = q;
}
void tform_fin(Tform *q) { UC(efree(q)); }

static void report(float a0[3], float a1[3],   float b0[3], float b1[3]) {
    enum {X, Y, Z};
    MSG("a0: %g %g %g\n", a0[X], a0[Y], a0[Z]);
    MSG("a1: %g %g %g\n", a1[X], a1[Y], a1[Z]);
    MSG("b0: %g %g %g\n", b0[X], b0[Y], b0[Z]);
    MSG("b1: %g %g %g\n", b1[X], b1[Y], b1[Z]);
}
enum {OK, ZERO};
static int os(float a0, float a1,   float b0, float b1, /**/ float *o, float *s) {
    /* origin, space */
    const float eps = 1e-8;
    float d;
    d = b0 - a0;
    if (-eps < d && d < eps) return ZERO;
    *o = (a1*b0-a0*b1)/(b0-a0);
    *s = (b1-a1)      /(b0-a0);
    return OK;
}
void tform_vector(float a0[3], float a1[3],   float b0[3], float b1[3], /**/ Tform* t) {
    int r;
    enum {X, Y, Z};
    r = os(a0[X], a1[X], b0[X], b1[X], /**/ &t->o[X], &t->s[X]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
        
    r = os(a0[Y], a1[Y], b0[Y], b1[Y], /**/ &t->o[Y], &t->s[Y]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
    
    r = os(a0[Z], a1[Z], b0[Z], b1[Z], /**/ &t->o[Z], &t->s[Z]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
}

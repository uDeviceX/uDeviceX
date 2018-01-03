#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "type.h"
#include "imp.h"

void tform_ini(Tform **pq) {
    Tform *q;
    UC(emalloc(sizeof(Tform), (void**)&q));
    *pq = q;
}
void tform_fin(Tform *q) { UC(efree(q)); }

static void report(float a0[3], float a1[3],   float b0[3], float b1[3]) {
    enum {X, Y, Z};
    MSG("a0: %g %g %g", a0[X], a0[Y], a0[Z]);
    MSG("a1: %g %g %g", a1[X], a1[Y], a1[Z]);
    MSG("b0: %g %g %g", b0[X], b0[Y], b0[Z]);
    MSG("b1: %g %g %g", b1[X], b1[Y], b1[Z]);
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

void tform_convert(Tform *t, float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

void tform_chain(Tform *t1, Tform *t2, /**/ Tform *t) {
    float a0[3] = {0, 0, 0};
    float b0[3] = {1, 1, 1};
    float a1[3], b1[3], a2[3], b2[3];

    tform_convert(t1, a0, /**/ a1);
    tform_convert(t1, b0, /**/ b1);

    tform_convert(t2, a1, /**/ a2);
    tform_convert(t2, b1, /**/ b2);

    UC(tform_vector(a0, a2, b0, b2, t));
}

void tform_inv(Tform *t1, /**/ Tform *t) {
    float a0[3] = {0, 0, 0};
    float b0[3] = {1, 1, 1};
    float a1[3], b1[3];

    tform_convert(t1, a0, /**/ a1);
    tform_convert(t1, b0, /**/ b1);

    UC(tform_vector(a1, a0,    b1, b0, t)); /* invert */
}

void tform_log(Tform *t) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    MSG("tform: o = [%g %g %g]", o[X], o[Y], o[Z]);
    MSG("tform: s = [%g %g %g]", s[X], s[Y], s[Z]);
}

void tform_dump(Tform *t, FILE *f) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    fprintf(f, "%16.10e %16.10e %16.10e\n", o[X], o[Y], o[Z]);
    fprintf(f, "%16.10e %16.10e %16.10e\n", s[X], s[Y], s[Z]);
}


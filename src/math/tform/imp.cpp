#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "type.h"
#include "imp.h"

void tform_ini(Tform **pq) {
    Tform *q;
    UC(emalloc(sizeof(Tform), (void**)&q));
    *pq = q;
}
void tform_fin(Tform *q) { UC(efree(q)); }

static void report(const float a0[3], const float a1[3],   const float b0[3], const float b1[3]) {
    enum {X, Y, Z};
    msg_print("a0: %g %g %g", a0[X], a0[Y], a0[Z]);
    msg_print("a1: %g %g %g", a1[X], a1[Y], a1[Z]);
    msg_print("b0: %g %g %g", b0[X], b0[Y], b0[Z]);
    msg_print("b1: %g %g %g", b1[X], b1[Y], b1[Z]);
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
void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform* t) {
    int r;
    enum {X, Y, Z};
    r = os(a0[X], a1[X], b0[X], b1[X], /**/ &t->o[X], &t->s[X]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Y], a1[Y], b0[Y], b1[Y], /**/ &t->o[Y], &t->s[Y]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Z], a1[Z], b0[Z], b1[Z], /**/ &t->o[Z], &t->s[Z]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
}

static int smallp(float s[3]) {
    enum {X, Y, Z};
    const float eps = 1e-12;
    int cx, cy, cz;
    cx = -eps < s[X] && s[X] < eps;
    cy = -eps < s[Y] && s[Y] < eps;
    cz = -eps < s[Z] && s[Z] < eps;
    return cx && cy && cz;
}
void tform_convert(Tform *t, const float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    if (smallp(s))
        ERR("tform_convert failed: s = [%g %g %g]\n", s[X], s[Y], s[Z]);
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

void tform_log(Tform *t) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    msg_print("tform: o = [%g %g %g]", o[X], o[Y], o[Z]);
    msg_print("tform: s = [%g %g %g]", s[X], s[Y], s[Z]);
}

void tform_dump(Tform *t, FILE *f) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    fprintf(f, "%16.10e %16.10e %16.10e\n", o[X], o[Y], o[Z]);
    fprintf(f, "%16.10e %16.10e %16.10e\n", s[X], s[Y], s[Z]);
}

static void to_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform* t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a0, a1,   b0, b1, /**/ t));
}
static void from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform* t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a1, a0,   b1, b0, /**/ t));
}
enum {LH_OK, LH_BAD};
static int assert_lh(const float lo[3], const float hi[3]) {
    enum {X, Y, Z};
    int c;
    c = lo[X] < hi[X] && lo[X] < hi[X] && lo[X] < hi[X];
    return c ? LH_OK : LH_BAD;
}
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3], /**/
                     Tform *t) {
    enum {X, Y, Z};
    /* f: from, t: to, g: global */
    Tform *f2g, *g2t;
    if (assert_lh(f_lo, f_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            f_lo[X], f_lo[Y], f_lo[X], f_hi[X], f_hi[Y], f_hi[Z]);

    if (assert_lh(t_lo, t_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            t_lo[X], t_lo[Y], t_lo[X], t_hi[X], t_hi[Y], t_hi[Z]);
    
    UC(tform_ini(&f2g));
    UC(tform_ini(&g2t));
    
    UC(from_grid(f_lo, f_hi, f_n, /**/ f2g));
    UC(to_grid  (t_lo, t_hi, t_n, /**/ g2t));
    UC(tform_chain(f2g, g2t, /**/ t));
    
    tform_fin(g2t);
    tform_fin(f2g);
}

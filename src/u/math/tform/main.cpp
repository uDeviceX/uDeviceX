#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "parser/imp.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/tform/imp.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "wall/sdf/tform/imp.h"

#include "tok.h"
#include "lib/imp.h"

struct TVec {
    float a0[3], b0[3];
    float a1[3], b1[3];
};

struct TGrid {
    float lo[3], hi[3];
    int n[3];
};

/* texture: */
struct TTex { int T[3], N[3], M[3]; };

struct TInput {
    TVec  v, u;
    TGrid f, t;
    TTex  tex;
};

static int Chain, Dev, Grid, Tex;
static Coords *coords;

static void usg0() {
    fprintf(stderr, "./udx -- OPTIONS.. < FILE\n");
    exit(0);
}

static void grid_log(TGrid *g) {
    enum {X, Y, Z};
    msg_print("lo: %g %g %g", g->lo[X], g->lo[Y], g->lo[Z]);
    msg_print("hi: %g %g %g", g->hi[X], g->hi[Y], g->hi[Z]);
    msg_print("n: %d %d %d", g->n[X], g->n[Y], g->n[Z]);
}

static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static float eatof(const char *s) {
    int n0;
    float v;
    n0 = sscanf(s, "%f", &v);
    if (n0 != 1) ERR("not a float: '%s'", s);
    return v;
}

static int eatoi(const char *s) {
    int n0;
    int v;
    n0 = sscanf(s, "%d", &v);
    if (n0 != 1) ERR("not an integer: '%s'", s);
    return v;
}

static void shift(int *c, char ***v) { (*c)--; (*v)++; }
static void shift_i(int *pc, char ***pv, int i) {
    /* delete argument v[i] */
    int c;
    char **v;
    c = *pc; v = *pv;
    for (; i + 1 < c; i++) v[i] = v[i + 1];
    c--;
    *pc = c; *pv = v;
}

static void assert_c(int c, const char *s) {
    if (c > 0) return;
    ERR("not enough args, %s", s);
}

enum {OK, END};
static int read(float *r) {
    enum {X, Y, Z};
    int n;
    n = scanf("%f %f %f", &r[X], &r[Y], &r[Z]);
    return (n == 3) ? OK : END;
}

static void convert(Tform *t, float a[3], /**/ float b[3]) {
    if (Dev) convert_dev(t, a, /**/ b);
    else     tform_convert(t, a, /**/ b);
}

static void process(Tform *t) {
    enum {X, Y, Z};
    float a[3], b[3];
    tform_log(t);
    while (read(/**/ a) == OK) {
        convert(t, a, /**/ b);
        printf("%g %g %g %g %g %g\n",
               a[X], a[Y], a[Z], b[X], b[Y], b[Z]);
    }
}

static void chain(TInput *v, Tform **pt) {
    Tform *t, *t1, *t2;
    t = *pt;

    UC(tform_ini(&t1));
    UC(tform_ini(&t2));
    UC(tform_vector(v->u.a0, v->u.a1,
                    v->u.b0, v->u.b1, /**/ t1));
    UC(tform_chain(t, t1, /**/ t2));
    tform_fin(t1);
    tform_fin(t);
    tform_log(t2);

    *pt = t2;
}

static void input2form(TInput *v, Tform **t) {
    TGrid *from, *to;
    TTex  *tex;
    if (Grid) {
        from = &v->f;
        to   = &v->t;
        grid_log(from);
        grid_log(to);
        tform_grid2grid(from->lo, from->hi, from->n,
                          to->lo, to->hi,   to->n, /**/ *t);
    } else if (Tex) {
        tex = &v->tex;
        tex2sdf_ini(coords, tex->T, tex->N, tex->M, /**/ *t);
    } else {
        UC(tform_vector(v->v.a0, v->v.a1,
                        v->v.b0, v->v.b1, /**/ *t));
        if (Chain) chain(v, t);
    }
}

static void main1(TInput *v) {
    Tform *t;
    UC(tform_ini(&t));
    UC(input2form(v, /**/ &t));
    UC(process(t));
    tform_fin(t);
}

static void read_float(int *pc, char ***pv, float *r) {
    enum {X, Y, Z};
    int c;
    char **v;

    c = *pc; v = *pv;

    assert_c(c, "X"); r[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "Y"); r[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "Z"); r[Z] = eatof(v[0]); shift(&c, &v);

    *pc = c; *pv = v;
}

static void read_int(int *pc, char ***pv, int *r) {
    enum {X, Y, Z};
    int c;
    char **v;

    c = *pc; v = *pv;

    assert_c(c, "X"); r[X] = eatoi(v[0]); shift(&c, &v);
    assert_c(c, "Y"); r[Y] = eatoi(v[0]); shift(&c, &v);
    assert_c(c, "Z"); r[Z] = eatoi(v[0]); shift(&c, &v);

    *pc = c; *pv = v;
}

static void read_vecs(int *c, char ***v, TVec *ve) {
    read_float(c, v, ve->a0);
    read_float(c, v, ve->a1);
    read_float(c, v, ve->b0);
    read_float(c, v, ve->b1);
}

static void read_grid(int *c, char ***v, TGrid *g) {
    read_float(c, v, g->lo);
    read_float(c, v, g->hi);
    read_int  (c, v, g->n);
}

static void read_tex(int *c, char ***v, TTex *t) {
    read_int(c, v, t->T);
    read_int(c, v, t->N);
    read_int(c, v, t->M);
}

static void main2(int c, char **v) {
    enum {X, Y, Z};
    TInput ve;
    if (Chain) {
        read_vecs(&c, &v, &ve.v);
        read_vecs(&c, &v, &ve.u);
    } else if (Grid) {
        read_grid(&c, &v, &ve.f);
        read_grid(&c, &v, &ve.t);
    } else if (Tex) {
        read_tex(&c, &v, &ve.tex);
    } else {
        read_vecs(&c, &v, &ve.v);
    }

    UC(main1(&ve));
}

static void usg(int c, char **v) {
    if (c > 0 && eq(v[0], "-h")) usg0();
}

static int flag(const char *a, int* pc, char ***pv) {
    int i, c, Flag;
    char **v;
    c = *pc; v = *pv;
    Flag = 0;
    for (i = 0; i < c; i++) {
        if (eq(a, v[i])) {
            shift_i(&c, &v, i);
            Flag = 1;
            break;
        }
    }
    *pc = c; *pv = v;
    return Flag;
}
int main(int argc, char **argv) {
    const char *arg;
    char **v;
    int c;
    const char delim[] = " \t";
    Config *cfg;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    coords_ini(m::cart, XS, YS, ZS, /**/ &coords);
    conf_ini(&cfg);
    conf_read(argc, argv, /**/ cfg);
    conf_lookup_string(cfg, "a", &arg);
    tok_ini(arg, delim, /**/ &c, &v);
    usg(c, v);
    Chain = flag("-c", &c, &v);
    Dev   = flag("-d", &c, &v);
    Grid  = flag("-g", &c, &v);
    Tex   = flag("-t", &c, &v);
    main2(c, v);

    tok_fin(c, v);
    coords_fin(coords);
    conf_fin(cfg);
    m::fin();
}

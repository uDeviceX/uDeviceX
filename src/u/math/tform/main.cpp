#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi/glb.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/tform/imp.h"

#include "utils/msg.h"

#include "lib/imp.h"

struct TVec {
    float a0[3], b0[3];
    float a1[3], b1[3];
};

struct TGrid {
    float lo[3], hi[3];
    int n[3];
};

struct TInput {
    TVec  v, u;
    TGrid f, t;
};

static int Inv, Chain, Dev, Grid;

static void usg0() {
    fprintf(stderr, "./udx -- OPTIONS.. < FILE\n");
    exit(0);
}

static void grid_log(TGrid *g) {
    enum {X, Y, Z};
    MSG("lo: %g %g %g", g->lo[X], g->lo[Y], g->lo[Z]);
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

static void inv(Tform **pt) {
    Tform *t1, *t2;
    t1 = *pt;

    UC(tform_ini(&t2));
    UC(tform_inv(t1, /**/ t2));
    UC(tform_fin(t1));
    tform_log(t2);

    *pt = t2;
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
    if (Grid) {
        from = &v->f;
        to   = &v->t;
        tform_grid2grid(from->lo, from->hi, from->n,
                          to->lo, to->hi,   to->n, /**/ *t);
    } else {
        UC(tform_vector(v->v.a0, v->v.a1,
                        v->v.b0, v->v.b1, /**/ *t));
        if (Chain) chain(v, t);
        if (Inv)   inv(t);
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
}

static void main2(int c, char **v) {
    enum {X, Y, Z};
    TInput ve;
    if (Chain) {
        read_vecs(&c, &v, &ve.v);
        read_vecs(&c, &v, &ve.u);
    } else if (Grid) {
        read_grid(&c, &v, &ve.f);
        read_grid(&c, &v, &ve.f);
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
    m::ini(&argc, &argv);
    usg(argc, argv);
    Inv   = flag("-i", &argc, &argv);
    Chain = flag("-c", &argc, &argv);
    Dev   = flag("-d", &argc, &argv);
    Grid  = flag("-g", &argc, &argv);
    main2(argc, argv);
    m::fin();
}

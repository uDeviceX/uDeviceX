#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi/glb.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/tform/imp.h"

#include "lib/imp.h"

struct TVec {
    float a0[3], b0[3];
    float a1[3], b1[3];

    float a2[3], b2[3];
    float a3[3], b3[3];
};

static int Inv, Chain, Dev;

static void usg0() {
    fprintf(stderr, "./udx -- OPTIONS.. < FILE\n");
    exit(0);
}
static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static float eatof(const char *s) {
    int n0;
    float v;
    n0 = sscanf(s, "%f", &v);
    if (n0 != 1) ERR("not a float: '%s'", s);
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

static void main0(Tform *t) {
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

static void chain(TVec *v, Tform **pt) {
    Tform *t, *t1, *t2;
    t = *pt;

    UC(tform_ini(&t1));
    UC(tform_ini(&t2));
    UC(tform_vector(v->a2, v->a3,   v->b2, v->b3, /**/ t1));
    UC(tform_chain(t, t1, /**/ t2));
    tform_fin(t1);
    tform_fin(t);
    tform_log(t2);

    *pt = t2;
}

static void main1(TVec *v) {
    Tform *t;
    UC(tform_ini(&t));
    UC(tform_vector(v->a0, v->a1,   v->b0, v->b1, /**/ t));
    if (Chain) chain(v, &t);
    if (Inv)   inv(&t);
    main0(t);
    tform_fin(t);
}

static void read_vec(int *pc, char ***pv, float *r) {
    enum {X, Y, Z};
    int c;
    char **v;

    c = *pc; v = *pv;

    assert_c(c, "X"); r[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "Y"); r[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "Z"); r[Z] = eatof(v[0]); shift(&c, &v);

    *pc = c; *pv = v;
}

static void main2(int c, char **v) {
    enum {X, Y, Z};
    TVec ve;
    read_vec(&c, &v, ve.a0);
    read_vec(&c, &v, ve.a1);
    read_vec(&c, &v, ve.b0);
    read_vec(&c, &v, ve.b1);
    if (Chain) {
        read_vec(&c, &v, ve.a2);
        read_vec(&c, &v, ve.a3);
        read_vec(&c, &v, ve.b2);
        read_vec(&c, &v, ve.b3);
    }
    main1(&ve);
}

static void usg(int c, char **v) {
    if (c > 0 && eq(v[0], "-h"))
        usg0();
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
    main2(argc, argv);
    m::fin();
}

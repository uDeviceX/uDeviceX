#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi/glb.h"
#include "msg.h"
#include "utils/error.h"
#include "math/tform/imp.h"

struct TVec {
    float a0[3], a1[3], b0[3], b1[3];
};

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

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
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

static void main0(Tform *t) {
    enum {X, Y, Z};
    float a[3], b[3];
    tform_log(t);
    while (read(/**/ a) == OK) {
        tform_convert(t, a, /**/ b);
        printf("%g %g %g %g %g %g\n",
               a[X], a[Y], a[Z], b[X], b[Y], b[Z]);
    }
}

static void main1(TVec *v) {
    Tform *t;
    float *a0, *a1, *b0, *b1;
    a0 = v->a0; a1 = v->a1; b0 = v->b0; b1 = v->b1;
    UC(tform_ini(&t));
    UC(tform_vector(a0, a1,   b0, b1, t));
    main0(t);
    tform_fin(t);
}

static void main2(int c, char **v) {
    enum {X, Y, Z};
    TVec ve;
    float *a0, *a1, *b0, *b1;
    a0 = ve.a0; a1 = ve.a1; b0 = ve.b0; b1 = ve.b1;
    assert_c(c, "a0[X]"); a0[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "a0[Y]"); a0[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "a0[Z]"); a0[Z] = eatof(v[0]); shift(&c, &v);

    assert_c(c, "a1[X]"); a1[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "a1[Y]"); a1[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "a1[Z]"); a1[Z] = eatof(v[0]); shift(&c, &v);

    assert_c(c, "b0[X]"); b0[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "b0[Y]"); b0[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "b0[Z]"); b0[Z] = eatof(v[0]); shift(&c, &v);

    assert_c(c, "b1[X]"); b1[X] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "b1[Y]"); b1[Y] = eatof(v[0]); shift(&c, &v);
    assert_c(c, "b1[Z]"); b1[Z] = eatof(v[0]); shift(&c, &v);

    main1(&ve);
}

static void usg(int c, char **v) {
    if (c > 0 && eq(v[0], "-h"))
        usg0();
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    usg(argc, argv);
    main2(argc, argv);
    m::fin();
}

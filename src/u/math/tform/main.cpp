#include <stdio.h>
#include <stdlib.h>

#include "mpi/glb.h"
#include "msg.h"
#include "utils/error.h"
#include "math/tform/imp.h"

struct TVec {
    float a0[3], a1[3], b0[3], b1[3];
};

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
}

static void assert_c(int c, const char *s) {
    if (c > 0) return;
    ERR("not enough args, %s", s);
}

static void main0(TVec *v) {
    Tform *t;
    float *a0, *a1, *b0, *b1;
    a0 = v->a0; a1 = v->a1; b0 = v->b0; b1 = v->b1;
    tform_ini(&t);
    tform_vector(a0, a1,   b0, b1, t);
    tform_dump(t, stdout);
    tform_fin(t);
}

static void main1(int c, char **v) {
    enum {X, Y, Z};
    TVec ve;
    float *a0, *a1, *b0, *b1;
    a0 = ve.a0; a1 = ve.a1; b0 = ve.b0; b1 = ve.b1;
    assert_c(c, "a0[X]"); a0[X] = atof(v[0]); shift(&c, &v);
    assert_c(c, "a0[Y]"); a0[Y] = atof(v[0]); shift(&c, &v);
    assert_c(c, "a0[Z]"); a0[Z] = atof(v[0]); shift(&c, &v);

    assert_c(c, "a1[X]"); a1[X] = atof(v[0]); shift(&c, &v);
    assert_c(c, "a1[Y]"); a1[Y] = atof(v[0]); shift(&c, &v);
    assert_c(c, "a1[Z]"); a1[Z] = atof(v[0]); shift(&c, &v);

    assert_c(c, "b0[X]"); b0[X] = atof(v[0]); shift(&c, &v);
    assert_c(c, "b0[Y]"); b0[Y] = atof(v[0]); shift(&c, &v);
    assert_c(c, "b0[Z]"); b0[Z] = atof(v[0]); shift(&c, &v);

    assert_c(c, "b1[X]"); b1[X] = atof(v[0]); shift(&c, &v);
    assert_c(c, "b1[Y]"); b1[Y] = atof(v[0]); shift(&c, &v);
    assert_c(c, "b1[Z]"); b1[Z] = atof(v[0]); shift(&c, &v);

    main0(&ve);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    main1(argc, argv);
    m::fin();
}

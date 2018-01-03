#include <stdio.h>
#include <stdlib.h>

#include "mpi/glb.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/linal/imp.h"

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
}

static void assert_c(int c, const char *s) {
    if (c > 0) return;
    ERR("not enough args, %s", s);
}

static void dump(float *r) {
    #define f "%10.6e"
    enum {XX, XY, XZ,   YY, YZ,  ZZ};
    printf(f " ", r[XX]);
    printf(f " ", r[XY]);
    printf(f " ", r[XZ]);
    printf(f " ", r[YY]);
    printf(f " ", r[YZ]);
    printf(f "\n", r[ZZ]);
    #undef f
}

static void main0(float *m) {
    float r[6];
    linal_inv3x3(m, /**/ r);
    dump(r);
}

static void main1(int c, char **v) {
    enum {XX, XY, XZ,   YY, YZ,  ZZ};
    float m[6];

    assert_c(c, "XX"); m[XX] = atof(v[0]); shift(&c, &v);
    assert_c(c, "XY"); m[XY] = atof(v[0]); shift(&c, &v);
    assert_c(c, "XZ"); m[XZ] = atof(v[0]); shift(&c, &v);
    assert_c(c, "YY"); m[YY] = atof(v[0]); shift(&c, &v);
    assert_c(c, "YZ"); m[YZ] = atof(v[0]); shift(&c, &v);
    assert_c(c, "ZZ"); m[ZZ] = atof(v[0]);

    main0(m);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    main1(argc, argv);
    m::fin();
}

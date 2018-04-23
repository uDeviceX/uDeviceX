#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <vector_types.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mesh/force/kantor1/imp.h"
#include "mesh/force/kantor0/imp.h"
#include "utils/mc.h"
#include "utils/imp.h"

static const char *prog = "udx";

typedef double3 (*Fun)(double, double, double3, double3, double3, double3);
enum {KANTOR0, KANTOR1};
static Fun Dih_A[] = {
    force_kantor0_hst::dih_a,
    force_kantor1_hst::dih_a,
};
static Fun Dih_B[] = {
    force_kantor0_hst::dih_b,
    force_kantor1_hst::dih_b,
};

void force(Fun fun,
           double phi, double kb,
           const double a0[3], const double b0[3], const double c0[3], const double d0[4], /**/
           double f0[3]) {
    enum {X, Y, Z};
    double3 a, b, c, d, f;
    a.x = a0[X]; a.y = a0[Y]; a.z = a0[Z];
    b.x = b0[X]; b.y = b0[Y]; b.z = b0[Z];
    c.x = c0[X]; c.y = c0[Y]; c.z = c0[Z];
    d.x = d0[X]; d.y = d0[Y]; d.z = d0[Z];
    UC(f = fun(phi, kb, a, b, c, d));
    f0[X] = f.x; f0[Y] = f.y; f0[Z] = f.z;
}

static void shift(int *c, char ***v) { (*c)--; (*v)++; }
void read_dbl(const char *v, double *x) {
    if (sscanf(v, "%lf", x) != 1)
        ERR("not a number '%s'\n", v);
}
void read3(int *pc, char ***pv, /**/ double a[3]) {
    enum {X, Y, Z};
    int c;
    char **v;
    c = *pc; v = *pv;
    if (c < 3) ERR("not enough arguments");
    UC(read_dbl(*v, /**/ &a[X])); shift(&c, &v);
    UC(read_dbl(*v, /**/ &a[Y])); shift(&c, &v);
    UC(read_dbl(*v, /**/ &a[Z])); shift(&c, &v);
    *pc = c; *pv = v;
}
void read1(int *pc, char ***pv, /**/ double *px) {
    int c;
    char **v;
    double x;
    c = *pc; v = *pv;
    if (c < 1) ERR("not enough arguments");
    UC(read_dbl(*v, /**/ &x)); shift(&c, &v);
    *px = x;
}

void read_type(int *pc, char ***pv, /**/ int *ptype) {
    int c, type;
    char **v;
    c = *pc; v = *pv;

    if (c < 0) ERR("not enough arguments");
    if      (same_str(v[0], "kantor0")) type = KANTOR0;
    else if (same_str(v[0], "kantor1")) type = KANTOR1;
    else ERR("unknown type '%s'", v[0]);
    shift(&c, &v);
    
    *pc = c; *pv = v; *ptype = type;
}

void main0(int *argc, char ***argv) {
    enum {X, Y, Z};
    int type;
    double phi, kb;
    double a[3], b[3], c[3], d[3], fa[3], fb[3];

    UC(read_type(argc, argv, &type));
    UC(read1(argc, argv, /**/ &phi));
    UC(read3(argc, argv, /**/ a));
    UC(read3(argc, argv, /**/ b));
    UC(read3(argc, argv, /**/ c));
    UC(read3(argc, argv, /**/ d));
    
    kb = 1;
    UC(force(Dih_A[type], phi, kb, a, b, c, d, /**/ fa));
    UC(force(Dih_B[type], phi, kb, a, b, c, d, /**/ fb));
    printf("%.16g %.16g %.16g %.16g %.16g %.16g\n",
           fa[X], fa[Y], fa[Z], fb[X], fb[Y], fb[Z]);
}

void usg() {
    fprintf(stderr, "%s [kantor0/kantor1] phi a[3] b[3] c[3] d[3]\n", prog);
    fprintf(stderr, "compute force on dihedral\n");
    exit(0);
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    UC(m::get_cart(MPI_COMM_WORLD, dims, &cart));

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    if (argc > 0 && same_str(argv[0], "-h")) usg();
    UC(main0(&argc, &argv));
    
    MC(m::Barrier(cart));
    m::fin();
}

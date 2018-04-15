#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "mesh/force/imp.h"
#include "utils/mc.h"

typedef double3 (*Fun)(double, double, double3, double3, double3, double3);

void force(Fun fun,
            double phi, double kb,
            const double a0[3], const double b0[3], const double c0[3], const double d0[4],
            double f0[3]) {
    enum {X, Y, Z};
    double3 a, b, c, d, f;
    a.x = a0[X]; a.y = a0[Y]; a.z = a0[Z];
    b.x = b0[X]; b.y = b0[Y]; b.z = b0[Z];
    c.x = c0[X]; c.y = c0[Y]; c.z = c0[Z];
    d.x = d0[X]; d.y = d0[Y]; d.z = d0[Z];
    f = fun(phi, kb, a, b, c, d);
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
    if (c < 4) ERR("not enough arguments");
    UC(read_dbl(*v, /**/ &a[X])); shift(&c, &v);
    UC(read_dbl(*v, /**/ &a[Y])); shift(&c, &v);
    UC(read_dbl(*v, /**/ &a[Z])); shift(&c, &v);
    
    *pc = c; *pv = v;
    
}

void main0(int *argc, char ***argv) {
    enum {X, Y, Z};
    double phi, kb;
    double a[3], b[3], c[3], d[3], fa[3], fb[3];

    shift(argc, argv); /* skip '--' */
    shift(argc, argv); /* skip '--' */
    
    UC(read3(argc, argv, /**/ a));
    UC(read3(argc, argv, /**/ b));
    UC(read3(argc, argv, /**/ c));
    UC(read3(argc, argv, /**/ d));
    
    phi = 0; kb = 1;
    force(force_hst::dih_a, phi, kb, a, b, c, d, /**/ fa);
    force(force_hst::dih_b, phi, kb, a, b, c, d, /**/ fb);
    printf("%.16g %.16g %.16g %.16g %.16g %.16g\n",
           fa[X], fa[Y], fa[Z], fb[X], fb[Y], fb[Z]);
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

    main0(&argc, &argv);
    
    MC(m::Barrier(cart));
    m::fin();
}

#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "mesh/scatter/imp.h"

#include "algo/scalars/imp.h"

static void dump_txt(int nv, int nm, const double *a) {
    int n, i;
    n = nv * nm;
    for (i = 0; i < n; i++)
        printf("%g\n", a[i]);
}

static void main0(const char *cell) {
    int nv, ne, nm;
    MeshRead *mesh;
    MeshScatter *scatter;
    Scalars *ones;
    double *rank;
    nm = 1;
    UC(mesh_read_ini_off(cell, /**/ &mesh));
    UC(mesh_scatter_ini(mesh, /**/ &scatter));
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);

    EMALLOC(nv, &rank);
    scalars_one_ini(ne, &ones);
    mesh_scatter_edg2vert(scatter, nm, ones, /**/ rank);
    dump_txt(nv, nm, rank);
    
    UC(mesh_scatter_fin(scatter));
    UC(mesh_read_fin(mesh));
    UC(scalars_fin(ones));
    EFREE(rank);
}

int main(int argc, char **argv) {
    const char *i;
    MPI_Comm comm;
    Config *cfg;
    int rank, size, dims[3];
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &comm);

    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "i", &i));

    main0(i);

    UC(conf_fin(cfg));
    MC(m::Barrier(comm));
    m::fin();
}

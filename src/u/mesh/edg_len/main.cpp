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
#include "mesh/vectors/imp.h"
#include "mesh/edg_len/imp.h"

#define PI (3.141592653589793)

static void main0(const char *path) {
    int i, nv, ne, nm;
    MeshRead *mesh;
    MeshEdgLen *len;
    Vectors  *pos;
    double *lens;
    UC(mesh_read_ini_off(path, /**/ &mesh));
    UC(mesh_edg_len_ini(mesh, &len));
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(mesh), /**/ &pos));

    nm = 1;
    EMALLOC(ne, &lens);
    mesh_angle_apply(len, nm, pos, /**/ lens);
    for (i = 0; i < ne; i++)
        printf("%g\n", lens);
    
    mesh_edg_len_fin(angle);
    UC(vectors_fin(pos));
    UC(mesh_read_fin(mesh));
    EFREE(angles);
}

int main(int argc, char **argv) {
    const char *i;
    Config *cfg;
    int rank, size, dims[3];
    MPI_Comm cart;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "i", &i));
    main0(i);

    UC(conf_fin(cfg));
    MC(m::Barrier(cart));
    m::fin();
}

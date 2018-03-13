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
#include "mesh/positions/imp.h"
#include "mesh/angle/imp.h"

void main0(const char *i) {
    int nv, ne, nm;
    MeshRead *mesh;
    MeshAngle *angle;
    Positions  *pos;
    double *angles;
    UC(mesh_read_ini_off(i, /**/ &mesh));
    UC(mesh_angle_ini(mesh, &angle));
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    UC(positions_float_ini(nv, mesh_read_get_vert(mesh), /**/ &pos));

    nm = 1;
    EMALLOC(ne, &angles);
    mesh_angle_apply(angle, nm, pos, /**/ angles);
    
    mesh_angle_fin(angle);
    UC(positions_fin(pos));
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

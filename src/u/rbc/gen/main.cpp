#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"
#include "rbc/matrices/imp.h"
#include "rbc/gen/imp.h"
#include "mesh/area/imp.h"
#include "mesh/positions/imp.h"
#include "inc/type.h"

#define MAX_N 99999
#define MAX_M 20
Particle pp[MAX_N];
double area[MAX_M];

void main0(const char *cell, const char *ic) {
    int i, nm, n, nv;
    Matrices *matrices;
    MeshRead *mesh;
    MeshArea *mesh_area;
    Positions *positions;
    const float *verts;
    UC(mesh_read_ini_off(cell, /**/ &mesh));
    UC(matrices_read(ic, &matrices));
    nv = mesh_read_get_nv(mesh);
    verts = mesh_read_get_vert(mesh);
    rbc_gen0(nv, verts, matrices, /**/ &n, pp);
    nm = n / nv;
    positions_particle_ini(n, pp, /**/ &positions);
    UC(mesh_area_ini(mesh, &mesh_area));
    mesh_area_apply(mesh_area, nm, positions, /**/ area);
    for (i = 0; i < nm; i++)
        printf("%g\n", area[i]);
    
    UC(positions_fin(positions));
    UC(mesh_area_fin(mesh_area));
    UC(matrices_fin(matrices));
    UC(mesh_read_fin(mesh));
}

int main(int argc, char **argv) {
    const char *cell;
    const char *ic;
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
    UC(conf_lookup_string(cfg, "cell", &cell));
    UC(conf_lookup_string(cfg, "ic", &ic));

    main0(cell, ic);

    UC(conf_fin(cfg));
    MC(m::Barrier(cart));
    m::fin();
}

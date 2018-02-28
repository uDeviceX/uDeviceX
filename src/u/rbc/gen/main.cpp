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

void main0(const char *cell, const char *ic) {
    Matrices *matrices;
    MeshRead *mesh;
    UC(mesh_read_ini_off(cell, /**/ &mesh));
    UC(matrices_read(ic, &matrices));

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

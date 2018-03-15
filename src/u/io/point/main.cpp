#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"
#include "coords/imp.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/mc.h"
#include "utils/imp.h"

#include "conf/imp.h"

#include "io/field/h5/imp.h"
#include "io/field/xmf/imp.h"

#include "mpi/wrapper.h"

void dump(MPI_Comm comm, const Coords *coords, const char *path) {
    enum {X, Y, Z};
    int rank;
    size_t nc;
    float *rho, *u[3];
    int sx, sy, sz;
    const char *names[] = { "density", "u", "v", "w" };

    MC(m::Comm_rank(comm, &rank));

    sx = xs(coords);
    sy = ys(coords);
    sz = zs(coords);
    
    nc = sx * sy * sz;
    EMALLOC(nc, &rho);
    EMALLOC(nc, &u[X]);
    EMALLOC(nc, &u[Y]);
    EMALLOC(nc, &u[Z]);
    
    float *data[] = { rho, u[X], u[Y], u[Z] };
    UC(h5_write(coords, comm, path, data, names, 4));    
    EFREE(rho); EFREE(u[X]); EFREE(u[Y]); EFREE(u[Z]);
    if (rank == 0) xmf_write(coords, path, names, 4);
}

void report(int i, int n, const char *path) {
    msg_print("write %s", path);
}

int main(int argc, char **argv) {
    const char *path;
    Coords *coords;
    Config *cfg;
    int i, ndump, rank, dims[3];
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    msg_ini(rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(coords_ini_conf(cart, cfg, &coords));

    UC(conf_lookup_string(cfg, "path", &path));
    UC(conf_lookup_int(cfg, "ndump", &ndump));
    
    for (i = 0; i < ndump; ++i) {
        report(i, ndump, path);
        dump(cart, coords, path);
    }

    UC(coords_fin(coords));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}

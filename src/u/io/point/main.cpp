#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/mc.h"
#include "utils/imp.h"

#include "conf/imp.h"
#include "io/point/imp.h"
#include "mpi/wrapper.h"

enum {MAX_N = 1000};

void main0(MPI_Comm comm, const char *path) {
    int id;
    IOPointConf *c;
    IOPoint *p;
    double rr[3*MAX_N];
    double density[MAX_N];

    UC(io_point_conf_ini(&c));
    UC(io_point_conf_push(c, "x y z"));
    UC(io_point_conf_push(c, "density"));
    UC(io_point_ini(MAX_N, path, c, &p));
    UC(io_point_conf_fin(c));

    UC(io_point_push(p, MAX_N, rr, "x y z"));
    UC(io_point_push(p, MAX_N, density, "density"));
    id = 0;
    UC(io_point_write(p, comm, id));
    UC(io_point_fin(p));
}

int main(int argc, char **argv) {
    const char *path;
    Config *cfg;
    int rank, dims[3];
    MPI_Comm comm;

    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &comm);

    MC(m::Comm_rank(comm, &rank));
    msg_ini(rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "o", &path));
    main0(comm, path);
    
    UC(conf_fin(cfg));
    MC(m::Barrier(comm));
    m::fin();
}

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

void main0(MPI_Comm, const char*) {
    IOPointConf *c;

    UC(io_point_conf_ini(&c));
    UC(io_point_conf_push(c, "x y z"));

    UC(io_point_conf_fin(c));
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
    UC(conf_fin(cfg));

    main0(comm, path);

    MC(m::Barrier(comm));
    m::fin();
}

#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "utils/error.h"
#include "utils/mc.h"

#include "xmf/imp.h"
#include "h5/imp.h"

static int3 get_total_size(int3 L, MPI_Comm cart) {
    enum {X, Y, Z, D};
    int dims[3], periods[D], coords[D];
    int3 G;
    MC(m::Cart_get(cart, D, dims, periods, coords));
    G.x = L.x * dims[X];
    G.y = L.y * dims[Y];
    G.z = L.z * dims[Z];
    return G;
}

void grid_write(int3 N, int3 L, MPI_Comm cart, const char *path, int ncmp, const float **data, const char **names) {
    int3 domainSize, gridSize;

    domainSize = get_total_size(L, cart);
    gridSize   = get_total_size(N, cart);
    
    UC(h5_write(N, cart, path, ncmp, data, names));
    if (m::is_master(cart))
        UC(xmf_write(domainSize, gridSize, path, ncmp, names));
}

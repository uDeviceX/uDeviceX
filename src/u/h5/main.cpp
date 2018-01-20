#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "glob/type.h"
#include "glob/ini.h"
#include "glob/imp.h"

#include "utils/msg.h"
#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/mc.h"
#include "utils/imp.h"

#include "io/field/h5/imp.h"
#include "io/field/xmf/imp.h"

#include "mpi/wrapper.h"

static int    argc;
static char **argv;

/* left shift */
void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "h5: not enough args\n");
        exit(2);
    }
}

void dump(MPI_Comm comm, const char *path, int sx, int sy, int sz) {
    enum {X, Y, Z};
    int rank;
    Coords coords;
    size_t size, nc;
    float *rho, *u[3];
    const char *names[] = { "density", "u", "v", "w" };

    UC(coords_ini(comm, /**/ &coords));
    MC(m::Comm_rank(comm, &rank));
    
    nc = sx * sy * sz;
    size = nc*sizeof(rho[0]);
    UC(emalloc(size, (void**) &rho));
    UC(emalloc(size, (void**) &u[X]));
    UC(emalloc(size, (void**) &u[Y]));
    UC(emalloc(size, (void**) &u[Z]));

    float *data[] = { rho, u[X], u[Y], u[Z] };
    UC(h5::write(coords, comm, path, data, names, 4));
    free(rho); free(u[X]); free(u[Y]); free(u[Z]);
    if (rank == 0) xmf_write(path, names, 4, sx, sy, sz);
    UC(coords_fin(/**/ &coords));
}

int ienv(const char *name, int def) {
    char *v;
    if ( (v = getenv(name))  == NULL ) return def;
    else return atoi(v);
}

void report(int i, int n, const char *path) {
    msg_print("write %s", path);
}

void main0(const char *path) {
    int ndump, i;
    int sx, sy, sz;
    
    ndump = ienv("ndump", 5);
    
    sx = XS; sy = YS; sz = ZS;
    for (i = 0; i < ndump; i++) {
        report(i, ndump, path);
        dump(m::cart, path, sx, sy, sz);
    }
}

void main1() {
    const char *path;
    path = argv[argc - 1]; lshift();
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    main0(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0; argv = argv0;
    main1();
}

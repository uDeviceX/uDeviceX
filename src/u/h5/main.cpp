#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"

#include "io/field/h5/imp.h"
#include "io/field/xmf/imp.h"

void dump(const char *path, int sx, int sy, int sz) {
    enum {X, Y, Z};
    size_t size, nc;
    float *rho, *u[3];
    const char *names[] = { "density", "u", "v", "w" };

    nc = sx * sy * sz;
    size = nc*sizeof(rho[0]);
    rho  = (float*)malloc(size);
    u[X] = (float*)malloc(size);
    u[Y] = (float*)malloc(size);
    u[Z] = (float*)malloc(size);

    float *data[] = { rho, u[X], u[Y], u[Z] };
    h5::write(path, data, names, 4, sx, sy, sz);
    free(rho); free(u[X]); free(u[Y]); free(u[Z]);
    if (m::rank == 0) xmf::write(path, names, 4, sx, sy, sz);
}

int ienv(const char *name, int def) {
    char *v;
    if ( (v = getenv(name))  == NULL ) return def;
    else return atoi(v);
}

void report(int i, int n, char *path) {
    if (n > 100 && i % 100 == 0)
        MSG("%06d/%06d: %s", i, n, path);
}

void get_path(int i, char *p) {
    sprintf(p, "i.h5");
}

void main0(int c, char **v) {
    int ndump, i;
    char path[BUFSIZ];
    int sx, sy, sz;
    
    ndump = ienv("ndump", 1000);
    sx = 4; sy = 8; sz = 16;
    for (i = 0; i < ndump; i++) {
        get_path(i, /**/ path);
        report(i, ndump, path);
        dump(path, sx, sy, sz);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0(argc, argv);
    m::fin();
}

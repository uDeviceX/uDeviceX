#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"

#include "io/field/h5/imp.h"
#include "io/field/xmf/imp.h"

void dump(const char *path, int nc) {
    enum {X, Y, Z};
    int sz;
    float *rho, *u[3];
    const char *names[] = { "density", "u", "v", "w" };

    nc = XS * YS * ZS;
    sz = nc*sizeof(rho[0]);
    rho  = (float*)malloc(sz);
    u[X] = (float*)malloc(sz);
    u[Y] = (float*)malloc(sz);
    u[Z] = (float*)malloc(sz);

    float *data[] = { rho, u[X], u[Y], u[Z] };
    h5::write(path, data, names, 4);
    free(rho); free(u[X]); free(u[Y]); free(u[Z]);
    if (m::rank == 0) xmf::write(path, names, 4);
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
    sprintf(p, "i.%06d.h5", i);
}

void main0(int c, char **v) {
    int n, i;
    char path[BUFSIZ];
    n = ienv("ndump", 1000);
    for (i = 0; i < n; i++) {
        get_path(i, /**/ path);
        report(i, n, path);
        dump(path, 32 * 32 * 32);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0(argc, argv);
    m::fin();
}

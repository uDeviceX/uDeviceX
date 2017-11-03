#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"

#include "io/field/h5/imp.h"
#include "io/field/xmf/imp.h"

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

void report(int i, int n, const char *path) {
    MSG("write %s", path);
}

void main0(const char *path) {
    int ndump, i;
    int sx, sy, sz;
    
    ndump = ienv("ndump", 5);
    
    sx = 4; sy = 8; sz = 16;
    for (i = 0; i < ndump; i++) {
        report(i, ndump, path);
        dump(path, sx, sy, sz);
    }
}

void main1() {
    const char *path;
    path = argv[argc - 1]; lshift();
    m::ini(argc, argv);
    main0(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0; argv = argv0;
    main1();
}

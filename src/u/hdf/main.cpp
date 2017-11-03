#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"

#include "io/field/h5/imp.h"

void dump(int nc) {
    enum {X, Y, Z};
    int sz;
    float *rho, *u[3];
    const char *names[] = { "density", "u", "v", "w" };

    sz = nc*sizeof(rho[0]);
    rho  = (float*)malloc(sz);
    u[X] = (float*)malloc(sz);
    u[Y] = (float*)malloc(sz);
    u[Z] = (float*)malloc(sz);

    float *data[] = { rho, u[X], u[Y], u[Z] };
    h5::write("main.h5", data, names, 4);
    free(rho); free(u[X]); free(u[Y]); free(u[Z]);
}

int ienv(const char *name, int def) {
    char *v;
    if ( (v = getenv(name))  == NULL ) return def;
    else return atoi(v);
}

void main0(int c, char **v) {
    int n, i;
    n = ienv("n", 1000);
    for (i = 0; i < n; i++) {
        printf(": %05d/%05d\n", i, n);
        dump(32 * 32 * 32);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0(argc, argv);
    m::fin();
}

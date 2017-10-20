#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "mc.h"

namespace mpicheck {
void check(int code, const char *file, int line) {
    char s[BUFSIZ];
    int n;
    if (code == MPI_SUCCESS) return;

    MPI_Error_string(code, /**/ s, &n);
    s[n + 1] = '\n';
    fprintf(stderr, "%s:%d: %s\n", file, line, s);
}
} // mpicheck

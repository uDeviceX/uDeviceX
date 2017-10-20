#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "mc.h"
#include "msg.h"

namespace mpicheck {
void check(int code, const char *file, int line) {
    char error_string[BUFSIZ];
    int sz;
    if (code != MPI_SUCCESS) return;
    MPI_Error_string(code, error_string, /**/ &sz);
    ERR("%s:%d: %s", file, line, error_string);
}
} // mpicheck

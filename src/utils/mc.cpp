#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "mc.h"

#include "utils/error.h"

namespace mpicheck {
void check(int code, const char *file, int line) {
    char msg[BUFSIZ];
    int n;
    if (code == MPI_SUCCESS) return;

    MPI_Error_string(code, /**/ msg, &n);
    UdxError::signal_mpi_error(file, line, msg);
    UdxError::report();
    UdxError::abort();
}
} // mpicheck

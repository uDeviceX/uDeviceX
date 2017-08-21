#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "mc.h"

namespace mpicheck {
void check(int code, const char *file, int line) {
    if (code != MPI_SUCCESS) {
        char error_string[2048];
        int length_of_error_string = sizeof(error_string);
        MPI_Error_string(code, error_string, &length_of_error_string);
        printf("mpiAssert: %s %d %s\n", file, line, error_string);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}
} // mpicheck

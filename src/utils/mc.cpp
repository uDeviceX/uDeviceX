#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "utils/error.h"
#include "mc.h"

namespace mpicheck {
void check(int code, const char *file, int line) {
    char msg[BUFSIZ], msg0[BUFSIZ];
    int n;
    if (m::is_success(code)) return;
    m::Error_string(code, /**/ msg0, &n);
    sprintf(msg, "%s (%d)", msg0, code);
    UdxError::signal_mpi_error(file, line, msg);
    UdxError::report();
    UdxError::abort();
}
} // mpicheck

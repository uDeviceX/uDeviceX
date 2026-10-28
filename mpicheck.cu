#include <stdio.h>
#include <mpi.h>

namespace m {
bool is_success(int errorcode);
int Error_string(int errorcode, char *string, int *resultlen);
}

void error_signal_mpi(const char *file, int line, const char *msg);
void error_report();
void error_abort();

namespace mpicheck {
void check(int code, const char *file, int line) {
    char msg[BUFSIZ], msg0[BUFSIZ];
    int n;
    if (m::is_success(code)) return;
    m::Error_string(code,      msg0, &n);
    sprintf(msg, "%s (%d)", msg0, code);
    error_signal_mpi(file, line, msg);
    error_report();
    error_abort();
}
}

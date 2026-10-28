#include "device.cuh"

void error_signal_cuda(const char *file, int line, const char *msg);
void error_report();
void error_abort();

namespace cc {
void check(int rc, const char *file, int line) {
    if (rc != 0) {
        error_signal_cuda(file, line, d::emsg());
        error_report();
        error_abort();
    }
}
}

#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "d/api.h"

#include "cc/common.h"
namespace cc {
void check(int rc, const char *file, int line) {
    if (rc != 0) {
        error_signal_cuda(file, line, d::emsg());
        error_report();
        error_abort();
    }
}
}

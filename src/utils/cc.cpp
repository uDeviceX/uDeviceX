#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "d/api.h"

#include "cc/common.h"
namespace cc {
void check(int rc, const char *file, int line) {
    if (rc != 0) {
        UdxError::signal_cuda_error(file, line, d::emsg());
        UdxError::report();
        UdxError::abort();
    }
}
}

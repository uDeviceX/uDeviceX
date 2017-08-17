#include <stdio.h>
#include "msg.h"
#include "cc/common.h"
namespace cc {
void check(cudaError_t rc, const char *file, int line) {
    if (rc != cudaSuccess)
        ERR("%s:%d: %s", file, line, cudaGetErrorString(rc));
}
}

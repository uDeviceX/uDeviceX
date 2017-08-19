#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"

#include "d/type.h"
#include "d/api.h"

#include "cc/common.h"
namespace cc {
void check(d::Error_t rc, const char *file, int line) {
    if (rc != cudaSuccess)
        ERR("%s:%d: %s", file, line, d::GetErrorString(rc));
}
}

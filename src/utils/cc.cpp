#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "d/api.h"

#include "cc/common.h"
namespace cc {
void check(int rc, const char *file, int line) {
    if (rc != 0) ERR("%s:%d: %s", file, line, d::emsg());
}
}

#include <stdio.h>
#include <assert.h>

#include "msg.h"
#include "error.h"

void err_handle0(const UdxError e, const char *file, const int line) {
    if (e.status == UdxError::SUCCESS) return;

    ERR("%s: %d: error caught from\n %s\n", file, line, e.msg);
}

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

#include "msg.h"
#include "error.h"

namespace UdxError {

static int err_line, err_status;
static char *err_file, err_msg[BUFSIZ];

void signal(const char *file, int line) {
    err_line = line;
    err_status = 1;
    strcpy(err_file, file);
    memset(err_msg, 0, sizeof(err_msg));
}

void signal_extra(const char *file, int line, const char *fmt, ...) {
    signal(file, line);
    va_list ap;
    va_start(ap, fmt);
    vsprintf(err_msg, fmt, ap);
    va_end(ap);    
}

void report(int line, const char *file) {
    ERR("%s: %d: Error: %s;  called from:\n"
        "%s: %d\n", err_file, err_line, err_msg, file, line);
}

} /* UdxError */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <string>

#include "msg.h"
#include "error.h"

namespace UdxError {

/* context information */
static int err_line, err_status = 0;
static const char *err_file;
static char err_msg[BUFSIZ];

/* context for MPI errors */
// TODO
static int mpi_status = 0; 

/* context for CUDA errors */
// TODO
static int cuda_status = 0;

/* stack used to dump backtrace in case of error */
enum {MAX_TRACE = 128};
static char stack[MAX_TRACE][BUFSIZ];
static int stack_sz = 0;
static char back_trace[MAX_TRACE * BUFSIZ];

static void stack_pop() {
    -- stack_sz;
    assert (stack_sz >= 0);
}

static void stack_push(const char *file, int line) {
    sprintf(stack[stack_sz], ": %s: %d:", file, line);
    ++ stack_sz;
    assert (stack_sz < MAX_TRACE);
}

static void stack_dump() {
    int i, nchar;
    char *bt = back_trace;
    
    for (i = 0; i < stack_sz; ++i) {
        nchar = sprintf(bt, "%s\n", stack[i]);
        assert(nchar >= 0);
        bt += nchar;
    }
}

void before(const char *file, int line) {
    stack_push(file, line);
}

void after() {
    stack_pop();
}

void signal(const char *file, int line) {
    err_line = line;
    err_status = 1;
    err_file = file;
    memset(err_msg, 0, sizeof(err_msg));
}

void signal_extra(const char *file, int line, const char *fmt, ...) {
    signal(file, line);
    va_list ap;
    va_start(ap, fmt);
    vsprintf(err_msg, fmt, ap);
    va_end(ap);    
}

bool error() {return err_status || mpi_status || cuda_status;}
void report(const char *file, int line) {
    if (err_status) {
        stack_dump();
        MSG("%s: %d: Error: %s\n"
            "backtrace:\n%s",
            err_file, err_line, err_msg, back_trace);
        exit(1);
    }
}

} /* UdxError */

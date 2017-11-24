#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <string>

#include "msg.h"
#include "error.h"

namespace UdxError {

/* context information */
static int         err_line;
static const char *err_file;
static char        err_msg [BUFSIZ];
static char        err_kind[BUFSIZ];
static int         err_status = 0;

/* stack used to dump backtrace in case of error */
enum {MAX_TRACE = 128};
static char stack      [ MAX_TRACE ][ BUFSIZ ];
static char back_trace [ MAX_TRACE  * BUFSIZ ];
static int  stack_sz = 0;


void stack_pop() {
    stack_sz--;
    assert (stack_sz >= 0);
}

void stack_push(const char *file, int line) {
    sprintf(stack[stack_sz], ": %s: %d:", file, line);
    stack_sz++;
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

static void set_err_loc(const char *file, int line) {
    err_line = line;
    err_file = file;
    memset(err_msg, 0, sizeof(err_msg));
}

void signal_error(const char *file, int line, const char *fmt, ...) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "udx");
    
    va_list ap;
    va_start(ap, fmt);
    vsprintf(err_msg, fmt, ap);
    va_end(ap);    
}

void signal_cuda_error(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "cuda");
    strcpy(err_msg, msg);
}

void signal_mpi_error(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "mpi");
    strcpy(err_msg, msg);
}

bool error() {return err_status;}
void report() {
    if (err_status) {
        stack_dump();
        MSG("%s: %d: %s error: %s\n"
            "backtrace:\n%s",
            err_file, err_line, err_kind, err_msg, back_trace);
    }
}

void abort() { exit(1); }

} /* UdxError */

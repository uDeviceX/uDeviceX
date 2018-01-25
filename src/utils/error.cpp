#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

#include "msg.h"
#include "error.h"

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


void error_stack_pop() {
    --stack_sz;
    assert (stack_sz >= 0);
}

static void check_stack_overflow() {
    if (stack_sz >= MAX_TRACE) {
        error_signal(__FILE__, __LINE__, "stack overflow (%d / %d)", stack_sz, MAX_TRACE);
        error_report();
        error_abort();
    }    
}

void error_stack_push(const char *file, int line) {
    sprintf(stack[stack_sz], ": %s: %d:", file, line);
    ++stack_sz;

    check_stack_overflow();
}

static void stack_dump() {
    int i, nchar;
    char *bt = back_trace;

    if (stack_sz) {
        nchar = sprintf(bt, "backtrace:\n");
        bt += nchar;
    }
    for (i = stack_sz-1; i >= 0; --i) {
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

void error_signal(const char *file, int line, const char *fmt, ...) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "udx");
    
    va_list ap;
    va_start(ap, fmt);
    vsprintf(err_msg, fmt, ap);
    va_end(ap);    
}

void error_signal_cuda(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "cuda");
    strcpy(err_msg, msg);
}

void error_signal_mpi(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "mpi");
    strcpy(err_msg, msg);
}

bool error_get() {return err_status;}
void error_report() {
    if (err_status) {
        stack_dump();
        msg_print("%s: %d: %s error: %s\n%s",
                  err_file, err_line, err_kind, err_msg, back_trace);
    }
}

void error_abort() { exit(1); }


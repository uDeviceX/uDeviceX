#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "msg.h"

static int rank;

void msg_ini(int rnk) {
    rank = rnk;
}

static FILE* open(const char *path) {
    static int fst = 1;
    if (fst) {
        fst = 0;
        return fopen(path, "w");
    } else {
        return fopen(path, "a");
    }
}

static void is_master(int r) {return r == 0;}
static void print(const char *msg, FILE *f) {
    fprintf(f, "%s\n", msg);
    if (is_master(rank))
        fprintf(stderr, ": %s\n", msg);
}

void msg_print(const char *fmt, ...) {
    char msg[BUFSIZ], name[64];
    va_list ap;
    FILE *f;

    // set name of file
    sprintf(name, ".%03d", rank);

    // form the message
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);

    // print the message
    f = open(n);
    print(f);
    fclose(f);
}


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "msg.h"

static int rank;

void msg_ini(int rnk) { rank = rnk; }

static FILE* open(const char *path) {
    static int fst = 1;
    FILE *f;
    if (fst) {
        fst = 0;
        f = fopen(path, "w");
        if (f == NULL) {
            fprintf(stderr, "%s:%d: fail to write: '%s'\n", __FILE__, __LINE__, path);
            exit(2);
        }
    } else {
        f = fopen(path, "a");
        if (f == NULL) {
            fprintf(stderr, "%s:%d: fail to append: '%s'\n", __FILE__, __LINE__, path);
            exit(2);
        }
    }
    return f;
}

static bool is_master(int r) {return r == 0;}
static void print(const char *msg, FILE *f) {
    fprintf(f, "%s\n", msg);
    if (is_master(rank))
        fprintf(stderr, ": %s\n", msg);
}

void msg_print(const char *fmt, ...) {
    char msg[BUFSIZ], name[FILENAME_MAX];
    va_list ap;
    FILE *f;
    snprintf(name, FILENAME_MAX - 1, ".%03d", rank);

    va_start(ap, fmt);
    vsnprintf(msg, BUFSIZ - 1, fmt, ap);
    va_end(ap);

    f = open(name);
    print(msg, f);
    if (fclose(f) != 0) {
        fprintf(stderr, "%s:%d: fail to close: '%s'\n", __FILE__, __LINE__, name);
        exit(2);
    }
}

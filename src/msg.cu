#include <stdio.h>
#include "m.h"

#include "msg.h"

namespace msg {
char buf[BUFSIZ];
static char fmt[] = ".%03d";

static FILE* safe_fopen(const char *path, const char *mode) {
    FILE *f;
    f = fopen(path, mode);
    if (!f) {
        fprintf(stderr, "fail to open: %s\n", path);
        exit(1);
    }
    return f;
}

static FILE* open(const char *path) {
    static int fst = 1;
    if (fst) {
        fst = 0;
        return safe_fopen(path, "w");
    } else
        return safe_fopen(path, "a");
}

static void print0(FILE *f) {
    fprintf(f, "%s\n", buf);
    if (m::rank == 0) fprintf(stderr, ": %s\n", buf);
}

void print() {
    char n[BUFSIZ]; /* name */
    FILE *f;
    sprintf(n, fmt, m::rank);
    f = open(n);
    print0(f);
    fclose(f);
}
}

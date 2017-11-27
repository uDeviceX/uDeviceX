#include <stdio.h>
#include <stdlib.h>

#include "mpi/glb.h"
#include "msg.h"

namespace msg {
char buf[BUFSIZ];
static char fmt[] = ".%03d";

static FILE* open(const char *path) {
    static int fst = 1;
    if (fst) {
        fst = 0;
        return fopen(path, "w");
    } else {
        return fopen(path, "a");
    }
}

static void print0(FILE *f) {
    fprintf(f, "%s\n", buf);
    if (m::rank == 0) fprintf(stderr, ": %s\n", buf);
}

void exit(int status) { ::exit(status); }
void print() {
    char n[BUFSIZ]; /* name */
    FILE *f;
    sprintf(n, fmt, m::rank);
    f = open(n);
    print0(f);
    fclose(f);
}
}

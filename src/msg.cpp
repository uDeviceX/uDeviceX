#include <stdio.h>
#include <stdlib.h>

#include "mpi/glb.h"
#include "msg.h"
#include "utils/error.h"
#include "utils/efopen.h"

namespace msg {
char buf[BUFSIZ];
static char fmt[] = ".%03d";

static FILE* open(const char *path) {
    static int fst = 1;
    FILE *f;
    if (fst) {
        fst = 0;
        UC(efopen(path, "w", /**/ &f));
    } else {
        UC(efopen(path, "a", /**/ &f));
    }
    return f;
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
    UC(efclose(f));
}
}

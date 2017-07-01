#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reader.h"

int separator(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
    if (strcmp("--", argv[i]) == 0) return i;
    return -1;
}

int main(int argc, char **argv) {

    if (argc < 6) {
        fprintf(stderr, "Usage: po.diffCoeff <inpp-*.bop> -- <inii-*.bop>\n");
        exit(1);
    }

    const int sep = separator(argc, argv);
    const int nin = sep - 1;
    char **ffpp = argv + 1;
    char **ffii = ffpp + sep;
    
    ReadData dpp0, dii0, *dpp, *dii;
    init(&dpp0); init(&dii0);

    for (int i = 0; i < nin; ++i) {
        printf("%s -- %s\n", ffpp[i], ffii[i]);
    }
    
    finalize(&dpp0); finalize(&dii0);
    return 0;
}

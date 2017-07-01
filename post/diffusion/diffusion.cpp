#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reader.h"

#define ERR(...) do {                            \
        fprintf(stderr,__VA_ARGS__);             \
        exit(1);                                 \
    } while (0);

int separator(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
    if (strcmp("--", argv[i]) == 0) return i;
    return -1;
}

void read_data(const char *fpp, ReadData *dpp, const char *fii, ReadData *dii) {
    read(fpp, dpp);
    read(fii, dii);

    if (dpp->type != FLOAT) ERR("expected float data form <%s>\n", fpp);
    if (dii->type != INT)   ERR("expected int   data form <%s>\n", fii);
}

int main(int argc, char **argv) {

    if (argc < 4) {
        fprintf(stderr, "Usage: po.diffCoeff <inpp-*.bop> -- <inii-*.bop>\n");
        exit(1);
    }

    const int sep = separator(argc, argv);
    const int nin = sep - 1;

    if (nin < 2) ERR("Need more than one file\n");
    
    char **ffpp = argv + 1;
    char **ffii = ffpp + sep;
    
    ReadData dpp0, dii0, dpp, dii;
    init(&dpp0); init(&dii0);

    read_data(ffpp[0], &dpp0, ffii[0], &dii0);
    
    for (int i = 1; i < nin; ++i) {
        init(&dpp);  init(&dii);
        read_data(ffpp[i], &dpp, ffii[i], &dii);
        finalize(&dpp);  finalize(&dii);
    }
    
    finalize(&dpp0); finalize(&dii0);
    return 0;
}

#include "stdio.h"
#include "stdlib.h"

#include "bop_reader.h"
extern "C" {
#include "bov.h"
}

struct Args {
    int lx, ly, lz;
    char *bop_s, *bop_c;
};

static void usg() {
    fprintf(stderr, "usg: u.color.density Lx Ly Lz <solvent.bop> <colors.bop>\n");
    exit(1);
}

static void parse(int argc, char **argv, /**/ Args *a) {
    if (argc != 6) usg();
    int iarg = 1;
    a->lx = atoi(argv[iarg++]);
    a->ly = atoi(argv[iarg++]);
    a->lz = atoi(argv[iarg++]);
    a->bop_s = argv[iarg++];
    a->bop_c = argv[iarg++];
}

static bool valid(int i, int l) {return i >= 0 && i < l;}

static void collect(long n, const float *pp, const int *cc, int lx, int ly, int lz, float *grid) {
    enum {X, Y, Z};
    long i, ix, iy, iz, cid;
    const float *r;
    for (i = 0; i < n; ++i) {
        r = pp + 6 * i;

        ix = (int) r[X];
        iy = (int) r[Y];
        iz = (int) r[Z];

        if (valid(ix, lx) && valid(iy, ly) && valid(iz, lz)) {
            cid = ix + lx * (iy + ly * iz);
            grid[cid] += cc[i];
        }
    }
}

int main(int argc, char **argv) {
    Args a;
    BopData bop_s, bop_c;
    BovDesc bov;

    parse(argc, argv, /**/ &a);

    init(&bop_s);
    init(&bop_c);

    read(a.bop_s, /**/ &bop_s);
    read(a.bop_c, /**/ &bop_c);

    summary(&bop_s);
    summary(&bop_c);
    
    bov.nx = a.lx; bov.ny = a.ly; bov.nz = a.lz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = 0;    bov.oy = 0;    bov.oz = 0;
    bov.data = NULL;
    sprintf(bov.var, "color density");
    bov.ncmp = 1;    

    finalize(&bop_s);
    finalize(&bop_c);

    return 0;
}

#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h" /* mini-MPI and -device */
#include "d/api.h"

#include "glb.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "utils/cc.h"

#include "algo/scan/int.h"
#include "clistx/imp.h"

enum {X,Y,Z};

#define MAXN 10000

void ini_1ppc(int3 d, int *n, Particle *pp) {
    int i, ix, iy, iz;
    Particle p;
    *n = d.x * d.y * d.z;

    for (i = 0, iz = 0; iz < d.z; ++iz)
        for (iy = 0; iy < d.y; ++iy)
            for (ix = 0; ix < d.x; ++ix) {
                p.r[X] = -d.x * 0.5f + ix + 0.5f;
                p.r[Y] = -d.y * 0.5f + iy + 0.5f;
                p.r[Z] = -d.z * 0.5f + iz + 0.5f;
                p.v[X] = p.v[Y] = p.v[Z] = 0.f;
                pp[i++] = p;
            }
}

void ini_random(int3 d, int density, int *n, Particle *pp) {
    Particle p;
    int i, N;
    N = *n = d.x * d.y * d.z * density;

    for (i = 0; i < N; ++i) {
        p.r[X] = (-0.5 + 0.999 * drand48()) * d.x;
        p.r[Y] = (-0.5 + 0.999 * drand48()) * d.y;
        p.r[Z] = (-0.5 + 0.999 * drand48()) * d.z;
        p.v[X] = p.v[Y] = p.v[Z] = 0.f;
        pp[i] = p;
    }
}

int3 ccoords(int3 d, int cid) {
    int3 c;
    c.x = cid % d.x;
    c.z = cid / (d.y * d.x);
    c.y = (cid - d.y * d.x * c.z) / d.x;
    return c;
}

bool valid(int c, int d, float x) {
    return (x >= c - 0.5 * d) && (x < c + 1 - 0.5 * d);
}

void verify_cell(int3 d, int cid, int s, int c, const Particle *pp) {
    int i, j;
    Particle p;
    int3 cell = ccoords(d, cid);
    for (i = 0; i < c; ++i) {
        j = s + i;
        p = pp[j];
        // MSG("%3f %3f %3f at %d %d %d",
        //     p.r[X], p.r[Y], p.r[Z], cell.x, cell.y, cell.z);
        assert(valid(cell.x, d.x, p.r[X]));
        assert(valid(cell.y, d.y, p.r[Y]));
        assert(valid(cell.z, d.z, p.r[Z]));
    }
}

void verify(int3 d, const int *starts, const int *counts, const Particle *pp, int n) {
    int cid, s, c, nc;
    nc = d.x * d.y * d.z;
    for (cid = 0; cid < nc; ++cid) {
        s = starts[cid];
        c = counts[cid];
        verify_cell(d, cid, s, c, pp);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);

    Particle *pplo, *ppre, *ppout;
    Particle *pp_hst;
    int nlo = 0, nre = 0, *starts, *counts, n;
    int3 dims = make_int3(4, 8, 4);
    clist::Clist clist;
    clist::Work work;

    ini(dims.x, dims.y, dims.z, /**/ &clist);
    ini_work(&clist, /**/ &work);

    pp_hst = (Particle*) malloc(MAXN * sizeof(Particle));
    counts = (int*) malloc(clist.ncells * sizeof(int));
    starts = (int*) malloc(clist.ncells * sizeof(int));
    CC(d::Malloc((void**) &pplo, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &ppre, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &ppout, MAXN * sizeof(Particle)));
       
    ini_1ppc(dims, /**/ &nlo, pp_hst);
    CC(d::Memcpy(pplo, pp_hst, nlo * sizeof(Particle), H2D));
    ini_random(dims, 4, /**/ &nre, pp_hst);
    CC(d::Memcpy(ppre, pp_hst, nre * sizeof(Particle), H2D));

    n = nlo + nre;
    
    build(nlo, nre, n, pplo, ppre, /**/ ppout, &clist, /*w*/ &work);
    
    CC(d::Memcpy(counts, clist.counts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(starts, clist.starts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(pp_hst, ppout, n * sizeof(Particle), D2H));
    
    verify(dims, starts, counts, pp_hst, n);    

    CC(d::Free(pplo));
    CC(d::Free(ppre));
    CC(d::Free(ppout));
    free(counts);
    free(starts);
    free(pp_hst);

    fin(/**/ &clist);
    fin_work(/**/ &work);

    
    m::fin();
}

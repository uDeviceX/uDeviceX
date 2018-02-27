#include <stdio.h>
#include <mpi.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "d/api.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "conf/imp.h"
#include "coords/ini.h"
#include "coords/imp.h"

#include "partlist/type.h"

#include "algo/scan/imp.h"
#include "clist/imp.h"

enum {X,Y,Z};

#define MAXN 10000

void read(int *n, Particle *pp) {
    int i;
    Particle p;
    i = 0;
    while (scanf("%f %f %f %f %f %f",
                 &p.r[X], &p.r[Y], &p.r[Z],
                 &p.v[X], &p.v[Y], &p.v[Z]) == 6)
        pp[i++] = p;
    *n = i;
}

int num_parts_inside(int3 L, int n, const Particle *pp) {
    enum {X, Y, Z};
    int i, ni;
    const float *r;
    for (i = ni = 0; i < n; ++i) {
        r = pp[i].r;
        ni +=
            (-L.x/2 <= r[X] && r[X] < L.x/2) &&
            (-L.y/2 <= r[Y] && r[Y] < L.y/2) &&
            (-L.z/2 <= r[Z] && r[Z] < L.z/2);            
    }
    return ni;
}

void print_cells(int3 L, const int *ss, const int *cc) {
    int i, n, s, c;
    n = L.x * L.y * L.z;

    for (i = 0; i < n; ++i) {
        s = ss[i];
        c = cc[i];
        printf("%d\n%d\n", s, c);
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

bool valid_cell(int3 d, int cid, int s, int c, const Particle *pp) {
    int i, j;
    Particle p;
    int3 cell = ccoords(d, cid);
    for (i = 0; i < c; ++i) {
        j = s + i;
        p = pp[j];
        // msg_print("%3f %3f %3f at %d %d %d",
        //     p.r[X], p.r[Y], p.r[Z], cell.x, cell.y, cell.z);
        if ( ! valid(cell.x, d.x, p.r[X]) ||
             ! valid(cell.y, d.y, p.r[Y]) ||
             ! valid(cell.z, d.z, p.r[Z])  )
            return false;
    }
    return true;
}

bool valid(int3 d, const int *starts, const int *counts, const Particle *pp, int n) {
    int cid, s, c, nc;
    nc = d.x * d.y * d.z;
    for (cid = 0; cid < nc; ++cid) {
        s = starts[cid];
        c = counts[cid];
        if (!valid_cell(d, cid, s, c, pp)) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    Particle *pp, *ppout;
    Particle *pp_hst;
    int n = 0, nout, *starts, *counts;
    int3 L;
    Clist clist;
    ClistMap *m;
    Config *cfg;
    Coords *coords;
    int rank, dims[3];
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    msg_ini(rank);
    
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(coords_ini_conf(cart, cfg, &coords));
    L = subdomain(coords);
    
    UC(clist_ini(L.x, L.y, L.z, /**/ &clist));

    EMALLOC(MAXN, &pp_hst);
    EMALLOC(clist.ncells, &counts);
    EMALLOC(clist.ncells, &starts);
    CC(d::Malloc((void**) &pp, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &ppout, MAXN * sizeof(Particle)));

    read(&n, pp_hst);
    nout = num_parts_inside(L, n, pp_hst);

    UC(clist_ini_map(n, 1, &clist, /**/ &m));
    
    CC(d::Memcpy(pp, pp_hst, n * sizeof(Particle), H2D));
    
    UC(clist_build(n, nout, pp, /**/ ppout, &clist, m));
    
    CC(d::Memcpy(counts, clist.counts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(starts, clist.starts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(pp_hst, ppout, nout * sizeof(Particle), D2H));
    
    if (valid(L, starts, counts, pp_hst, nout))
        printf("0\n");
    else
        printf("1\n");

    print_cells(L, starts, counts);
    
    CC(d::Free(pp));
    CC(d::Free(ppout));
    EFREE(counts);
    EFREE(starts);
    EFREE(pp_hst);
    
    UC(clist_fin(/**/ &clist));
    UC(clist_fin_map(/**/ m));
    UC(conf_fin(cfg));
    UC(coords_fin(coords));

    MC(m::Barrier(cart));
    m::fin();
}

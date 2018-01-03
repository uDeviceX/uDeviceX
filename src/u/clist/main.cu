#include <stdio.h>
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

#include "partlist/imp.h"

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
    msg_ini(0);    
    Particle *pp, *ppout;
    Particle *pp_hst;
    int n = 0, *starts, *counts;
    int3 dims;
    clist::Clist clist;
    clist::Map m;

    dims.x = XS;
    dims.y = YS;
    dims.z = ZS;
    
    ini(dims.x, dims.y, dims.z, /**/ &clist);

    UC(emalloc(MAXN * sizeof(Particle), (void**) &pp_hst));
    UC(emalloc(clist.ncells * sizeof(int), (void**) &counts));
    UC(emalloc(clist.ncells * sizeof(int), (void**) &starts));
    CC(d::Malloc((void**) &pp, MAXN * sizeof(Particle)));
    CC(d::Malloc((void**) &ppout, MAXN * sizeof(Particle)));

    read(&n, pp_hst);

    ini_map(n, 1, &clist, /**/ &m);
    
    CC(d::Memcpy(pp, pp_hst, n * sizeof(Particle), H2D));
    
    build(n, n, pp, /**/ ppout, &clist, &m);
    
    CC(d::Memcpy(counts, clist.counts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(starts, clist.starts, clist.ncells * sizeof(int), D2H));
    CC(d::Memcpy(pp_hst, ppout, n * sizeof(Particle), D2H));
    
    if (valid(dims, starts, counts, pp_hst, n))
        printf("0\n");
    else
        printf("1\n");

    print_cells(dims, starts, counts);
    
    CC(d::Free(pp));
    CC(d::Free(ppout));
    free(counts);
    free(starts);
    free(pp_hst);

    fin(/**/ &clist);
    fin_map(/**/ &m);
}

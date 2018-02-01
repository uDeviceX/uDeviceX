#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"

#include "frag/imp.h"
#include "comm/imp.h"

#include "inc/type.h"

#include "utils/error.h"
#include "utils/msg.h"

static void shift(int3 L, int fid, float r[3]) {
    enum {X, Y, Z};
    int d[3];
    fraghst::i2d3(fid, d);
    r[X] += d[X] * L.x;
    r[Y] += d[Y] * L.y;
    r[Z] += d[Z] * L.z;
}

static bool is_inside(int3 LW, const Particle p) {
    enum {X, Y, Z};
    return
        p.r[X] >= -0.5 * LW.x && p.r[X] < 0.5 * LW.x &&
        p.r[Y] >= -0.5 * LW.y && p.r[Y] < 0.5 * LW.y &&
        p.r[Z] >= -0.5 * LW.z && p.r[Z] < 0.5 * LW.z;
}

static void fill_bags(int3 L, int n, const Particle *pp, hBags *b) {
    int i, j, *cc, c;
    Particle p0, p, **dst;
    int3 LW;
    LW.x = L.x + 2*XWM;
    LW.y = L.y + 2*YWM;
    LW.z = L.z + 2*ZWM;

    cc  = b->counts;
    dst = (Particle **) b->data;

    memset(cc, 0, NBAGS * sizeof(int));
    
    for (i = 0; i < n; ++i) {
        p0 = pp[i];
        for (j = 0; j < NFRAGS; ++j) {
            p = p0;
            shift(L, j, p.r);
            if (is_inside(LW, p)) {
                c = cc[j] ++;
                dst[j][c] = p;
            }
        }
    }
}

static void communicate(const hBags *s, Comm *c, hBags *r) {
    UC(comm_post_send(s, c));
    UC(comm_post_recv(r, c));
    UC(comm_wait_send(c));
    UC(comm_wait_recv(c, /**/ r));
}

static void check_counts(int maxn, int n0, const hBags *b) {
    int i, c = n0;
    for (i = 0; i < NFRAGS; ++i) c += b->counts[i];
    if (c > maxn)
        ERR("Too many particles for wall : %d / %d\n", c, maxn);
}

static void unpack(int maxn, const hBags *b, /*io*/ int *n, Particle *pp) {
    int i, j, k, c;
    const Particle *src;
    k = *n;
    for (j = 0; j < NFRAGS; ++j) {
        c = b->counts[j];
        src = (const Particle *) b->data[j];
        for (i = 0; i < c; ++i) {
            pp[k] = src[i];
            ++k;
        }
    }
    *n = k;
}

/* exchange pp(hst) between processors to get a wall margin */
void wall_exch_pp(MPI_Comm cart, int3 L, int maxn, /*io*/ Particle *pp, int *n) {
    hBags send, recv;
    Comm *com;
    int i, capacity[NBAGS];

    for (i = 0; i < NBAGS; ++i) capacity[i] = maxn;
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &send, NULL));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &recv, NULL));
    UC(comm_ini(cart, &com));

    fill_bags(L, *n, pp, /**/ &send);
    communicate(&send, /**/ com, &recv);
    check_counts(maxn, *n, &recv);
    unpack(maxn, &recv, /**/ n, pp);
    
    UC(comm_bags_fin(HST_ONLY, NONE, &send, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, &recv, NULL));
    UC(comm_fin(com));
}

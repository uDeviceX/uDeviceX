#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "frag/imp.h"
#include "comm/imp.h"

#include "inc/type.h"

#include "utils/error.h"
#include "utils/msg.h"

using namespace comm;

enum {
    LX = XS + 2 * XWM,
    LY = YS + 2 * YWM,
    LZ = ZS + 2 * ZWM
};

static void shift(int fid, float r[3]) {
    enum {X, Y, Z};
    const int d[3] = frag_i2d3(fid);
    r[X] += d[X] * XS;
    r[Y] += d[Y] * YS;
    r[Z] += d[Z] * ZS;
}

static bool is_inside(const Particle p) {
    enum {X, Y, Z};
    return
        p.r[X] >= -0.5 * LX && p.r[X] < 0.5 * LX &&
        p.r[Y] >= -0.5 * LY && p.r[Y] < 0.5 * LY &&
        p.r[Z] >= -0.5 * LZ && p.r[Z] < 0.5 * LZ;
}

static void fill_bags(int n, const Particle *pp, hBags *b) {
    int i, j, *cc, c;
    Particle p0, p, **dst;

    cc  = b->counts;
    dst = (Particle **) b->data;

    memset(cc, 0, NBAGS * sizeof(int));
    
    for (i = 0; i < n; ++i) {
        p0 = pp[i];
        for (j = 0; j < NFRAGS; ++j) {
            p = p0;
            shift(j, p.r);
            if (is_inside(p)) {
                c = cc[j] ++;
                dst[j][c] = p;
            }
        }
    }
}

static void communicate(const hBags *s, Stamp *c, hBags *r) {
    UC(post_send(s, c));
    post_recv(r, c);
    wait_send(c);
    UC(wait_recv(c, /**/ r));
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
void exch(int maxn, /*io*/ Particle *pp, int *n) {
    hBags send, recv;
    Stamp stamp;
    int i, capacity[NBAGS];

    for (i = 0; i < NBAGS; ++i) capacity[i] = maxn;
    UC(ini(HST_ONLY, NONE, sizeof(Particle), capacity, &send, NULL));
    UC(ini(HST_ONLY, NONE, sizeof(Particle), capacity, &recv, NULL));
    UC(ini(m::cart, &stamp));

    fill_bags(*n, pp, /**/ &send);
    communicate(&send, /**/ &stamp, &recv);
    check_counts(maxn, *n, &recv);
    unpack(maxn, &recv, /**/ n, pp);
    
    fin(HST_ONLY, NONE, &send, NULL);
    fin(HST_ONLY, NONE, &recv, NULL);
    fin(&stamp);
}

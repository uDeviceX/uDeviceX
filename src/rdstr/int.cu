#include <cstdio>

#include "common.h"
#include "common.cuda.h"
#include "common.tmp.h"

#include "minmax.h"

//#include "mdstr/int.h"
#include "rdstr/int.h"

namespace rdstr {

enum {X, Y, Z};

void alloc_ticketE(/**/ TicketE *t) {
    t->ll = new PinnedHostBuffer2<float3>;
    t->hh = new PinnedHostBuffer2<float3>;
    t->rr = new float[3 * MAX_CELL_NUM];
}

void free_ticketE(/**/ TicketE *t) {
    delete t->ll;
    delete t->hh;
    delete[] t->rr;
}

static void lh2r(int n, const float3 *ll, const float3 *hh, /**/ float *rr) {
    for (int i = 0; i < n; ++i) {
        float3 l = ll[i], h = hh[i];
        float *r = rr + 3 * i;
        r[X] = 0.5f * (l.x + h.x);
        r[Y] = 0.5f * (l.y + h.y);
        r[Z] = 0.5f * (l.z + h.z);
    }
}

void get_pos(const Particle *pp, int nc, int nv, /**/ TicketE *t) {
    minmax(pp, nv, nc, /**/ t->ll->DP, t->hh->DP);
    dSync();
    lh2r(nc, t->ll->D, t->hh->D, /**/ t->rr);
}

} // rdstr

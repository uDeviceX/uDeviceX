#include <cstdio>
#include <mpi.h>

#include "common.h"
#include "common.cuda.h"
#include "common.tmp.h"

#include "basetags.h"

#include "mdstr/int.h"
#include "rdstr/int.h"
#include "rdstr/imp.h"

namespace rdstr {

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

void extents(const Particle *pp, int nc, int nv, /**/ TicketE *t) {
    t->ll->resize(nc);
    t->hh->resize(nc);
    sub::extents(pp, nc, nv, /**/ t->ll->DP, t->hh->DP);
}
    
void get_pos(int nc, /**/ TicketE *t) {
    dSync(); // wait for extents
    sub::get_pos(nc, t->ll->D, t->hh->D, /**/ t->rr);
}

} // rdstr

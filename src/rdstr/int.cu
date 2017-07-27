#include <cstdio>

#include "common.h"
#include "common.cuda.h"
#include "common.tmp.h"

//#include "mdstr/int.h"
#include "rdstr/int.h"

namespace rdstr {

void alloc_TicketE(/**/ TicketE *t) {
    t->ll = new PinnedHostBuffer2<float3>;
    t->hh = new PinnedHostBuffer2<float3>;
    t->rr = new float[3 * MAX_CELL_NUM];
}

void free_TicketE(/**/ TicketE *t) {
    delete t->ll;
    delete t->hh;
    delete[] t->rr;
}

void get_pos(const Particle *pp, int nc, int nv, /**/ TicketE *t) {
    
}

} // rdstr

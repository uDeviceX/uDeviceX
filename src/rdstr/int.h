namespace rdstr {
//using namespace mdstr;

/* ticket extend */
struct TicketE {
    PinnedHostBuffer2<float3> *ll, *hh;    /* extents of RBCs                   */
    float *rr;                             /* positions used to distribute rbcs */
};

void alloc_ticketE(/**/ TicketE *t);
void free_ticketE(/**/ TicketE *t);

void get_pos(const Particle *pp, int nc, int nv, /**/ TicketE *t);
}

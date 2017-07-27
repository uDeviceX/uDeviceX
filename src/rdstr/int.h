namespace rdstr {
//using namespace mdstr;

/* ticket extend */
struct TicketE {
    PinnedHostBuffer2<float3> *llo, *hhi;    /* extents of RBCs                   */
    float *rr;                               /* positions used to distribute rbcs */
};

void get_pos(const Particle *pp, /**/ TicketE *t);
}

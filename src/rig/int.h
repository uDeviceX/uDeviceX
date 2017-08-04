namespace rig {

struct Quants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Solid *ss_hst, *ss;
    float *rr0_hst, *rr0;
    Mesh m_hst, m_dev;
    Particle *i_pp_hst, *i_pp;

    Solid *ss_dmp;
};

struct TicketBB {
    float3 *minbb_hst, *maxbb_hst; /* [b]ounding [b]oxes of solid mesh on host   */
    float3 *minbb_dev, *maxbb_dev; /* [b]ounding [b]oxes of solid mesh on device */
    Solid *ss_hst, *ss;
    Particle *i_pp_hst, *i_pp;

    Solid *ss_dmp;
};

void alloc_quants(Quants *q);
void free_quants(Quants *q);

void alloc_ticket(TicketBB *t);
void free_ticket(TicketBB *t);

void gen_quants(Particle *opp, int *on, Quants *q);
void strt_quants(const int id, Quants *q);

void set_ids(Quants q);

void strt_dump_templ(const Quants q);
void strt_dump(const int id, const Quants q);

} // rig

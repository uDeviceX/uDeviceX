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
    int *tcs_hst, *tcc_hst, *tci_hst; /* [t]riangle cell-lists on host   */
    int *tcs_dev, *tcc_dev, *tci_dev; /* [t]riangle cell-lists on device */

    Solid *ss_dmp;
};

void alloc_quants(Quants *q) {
    q->n = q->ns = q->nps = 0;
    
    CC(cudaMalloc(&q->pp ,     MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&q->ss ,       MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&q->rr0, 3 * MAX_PART_NUM * sizeof(float)));
    CC(cudaMalloc(&q->i_pp,    MAX_PART_NUM * sizeof(Particle)));
    
    q->pp_hst   = new Particle[MAX_PART_NUM];
    q->ss_hst   = new Solid[MAX_SOLIDS];
    q->rr0_hst  = new float[3 * MAX_PART_NUM];
    q->i_pp_hst = new Particle[MAX_PART_NUM];

    q->ss_dmp = new Solid[MAX_SOLIDS];
}

void free_quants(Quants *q) {
    delete[] q->pp_hst;
    delete[] q->ss_hst;
    delete[] q->rr0_hst;
    delete[] q->i_pp_hst;
    
    CC(cudaFree(q->pp));
    CC(cudaFree(q->ss));
    CC(cudaFree(q->rr0));
    CC(cudaFree(q->i_pp));

    if (q->m_hst.tt) delete[] q->m_hst.tt;
    if (q->m_hst.vv) delete[] q->m_hst.vv;

    if (q->m_dev.tt) CC(cudaFree(q->m_dev.tt));
    if (q->m_dev.vv) CC(cudaFree(q->m_dev.vv));

    delete[] q->ss_dmp;
}

void alloc_ticket(TicketBB *t) {
    CC(cudaMalloc(&t->minbb_dev, MAX_SOLIDS * sizeof(float3)));
    CC(cudaMalloc(&t->maxbb_dev, MAX_SOLIDS * sizeof(float3)));
    CC(cudaMalloc(&t->i_pp,  MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&t->ss ,        MAX_SOLIDS * sizeof(Solid)));

    // assume 1 triangle don't overlap more than 27 cells
    CC(cudaMalloc(&t->tcs_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&t->tcc_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&t->tci_dev, 27 * MAX_SOLIDS * MAX_FACE_NUM * sizeof(int)));
    
    t->minbb_hst = new float3[MAX_SOLIDS];
    t->maxbb_hst = new float3[MAX_SOLIDS];
    t->ss_hst   = new Solid[MAX_SOLIDS];
    t->i_pp_hst = new Particle[MAX_PART_NUM];

    t->tcs_hst = new int[XS * YS * ZS];
    t->tcc_hst = new int[XS * YS * ZS];
    t->tci_hst = new int[27 * MAX_SOLIDS * MAX_FACE_NUM];

    t->ss_dmp = new Solid[MAX_SOLIDS];
}

void free_ticket(TicketBB *t) {
    CC(cudaFree(t->minbb_dev));
    CC(cudaFree(t->maxbb_dev));
    CC(cudaFree(t->i_pp));
    CC(cudaFree(t->ss));

    CC(cudaFree(t->tcs_dev));
    CC(cudaFree(t->tcc_dev));
    CC(cudaFree(t->tci_dev));

    delete[] t->minbb_hst;
    delete[] t->maxbb_hst;
    delete[] t->ss_hst;
    delete[] t->i_pp_hst;

    delete[] t->tcs_hst;
    delete[] t->tcc_hst;
    delete[] t->tci_hst;

    delete[] t->ss_dmp;
}

void create(Particle *opp, int *on, Quants *q) {
    sub::create(opp, on, &q->m_dev, &q->m_hst, &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst, q->pp_hst);
}

void gen_pp_hst(Quants q) {
    sub::gen_pp_hst(q.ns, q.rr0_hst, q.nps, /**/ q.ss_hst, q.pp_hst);
}

void gen_ipp_hst(const Quants q, TicketBB t) {
    sub::gen_ipp_hst(q.ss_hst, q.ns, q.m_hst, /**/ t.i_pp_hst);
}

void cpy_H2D(Quants q, TicketBB t) {
    cH2D(t.i_pp, t.i_pp_hst, q.ns * q.m_hst.nv);
    cH2D(q.ss,   q.ss_hst,   q.ns);
    cH2D(q.rr0,  q.rr0_hst,  q.nps * 3);
    cH2D(q.pp,   q.pp_hst,   q.n);
}

void set_ids(Quants q) {
    sub::ic::set_ids(q.ns, q.ss_hst);
}

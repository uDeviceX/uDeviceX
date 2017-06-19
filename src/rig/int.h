struct Quants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Solid *ss_hst, *ss;
    float *rr0_hst, *rr0;
    Mesh m_hst, m_dev;
};

struct TicketBB {
    float3 *minbb_hst, *maxbb_hst; /* [b]ounding [b]oxes of solid mesh on host   */
    float3 *minbb_dev, *maxbb_dev; /* [b]ounding [b]oxes of solid mesh on device */

    Solid *ss;
};

void alloc_quants(Quants *q) {
    n - ns = nps = 0;
    
    CC(cudaMalloc(&q->pp ,     MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&q->ss ,       MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&q->rr0, 3 * MAX_PART_NUM * sizeof(float)));
    
    q->pp_hst  = new Particle[MAX_PART_NUM];
    q->ss_hst  = new Solid[MAX_SOLIDS];
    q->rr0_hst = new float[3 * MAX_PART_NUM];
}

void free_quants(Quants *q) {
    delete[] q->pp_hst;
    delete[] q->ss_hst;
    delete[] q->rr0_hst;

    CC(cudaFree(q->pp));
    CC(cudaFree(q->ss));
    CC(cudaFree(q->rr0));

    if (m_hst.tt) delete[] m_hst.tt;
    if (m_hst.vv) delete[] m_hst.vv;

    if (m_dev.tt) CC(cudaFree(m_dev.tt));
    if (m_dev.vv) CC(cudaFree(m_dev.vv));
}

void alloc_ticket(TicketBB *t) {
    CC(cudaMalloc(&t->minbb_dev, MAX_SOLIDS * sizeof(float3)));
    CC(cudaMalloc(&t->maxbb_dev, MAX_SOLIDS * sizeof(float3)));

    t->minbb_hst = new float3[MAX_SOLIDS];
    t->maxbb_hst = new float3[MAX_SOLIDS];
}

void free_ticket(TicketBB *t) {
    CC(cudaFree(t->minbb_dev));
    CC(cudaFree(t->maxbb_dev));

    delete[] t->minbb_hst;
    delete[] t->maxbb_hst;
}


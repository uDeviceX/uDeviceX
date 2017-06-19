struct Quants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Solid *ss_hst, *ss;
    float *rr0_hst, *rr0;
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
}

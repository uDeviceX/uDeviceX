struct Quants {
    int n, nc, nt, nv;
    Particle *pp, *pp_hst;
    Texo <float2> texvert;       /* vertices */

    int *adj0, *adj1;            /*adjacency lists */
    Texo <int> texadj0, texadj1;

    int4 *tri;                   /* triangles */
    Texo <int4> textri;

    int *tri_hst;
    float *av;
};


void alloc_quants(Quants *q) {
    q->n = q->nc = 0;
    mpDeviceMalloc(&q->pp);
    mpHostMalloc(&q->pp_hst);

    q->tri = NULL;
    q->adj0 = q->adj1 = NULL;

    q->nt = RBCnt;
    q->nv = RBCnv;

    CC(cudaMalloc(&q->tri, q->nt * sizeof(int4)));
    CC(cudaMalloc(&q->adj0, q->nv * RBCmd * sizeof(int)));
    CC(cudaMalloc(&q->adj1, q->nv * RBCmd * sizeof(int)));
                      
    q->tri_hst = new int[MAX_FACE_NUM];
    CC(cudaMalloc(&q->av, MAX_CELL_NUM));
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp));
    CC(cudaFree(q->av));

    if (q->tri)  CC(cudaFree(q->tri));
    if (q->adj0) CC(cudaFree(q->adj0));
    if (q->adj1) CC(cudaFree(q->adj1));

    q->textri.destroy();
    q->texadj0.destroy();
    q->texadj1.destroy();
    q->texvert.destroy();
}

void setup(Quants q) {
    sub::setup(q.tri_hst, q.tri, &q.textri, q.adj0, &q.texadj0, q.adj1, &q.texadj1, q.pp, &q.texvert, q.n);
}

void forces(Quants q, /**/ Force *ff) {
    sub::forces(q.nc, q.texvert, q.textri, q.texadj0, q.texadj1, /**/ ff, q.av);
}

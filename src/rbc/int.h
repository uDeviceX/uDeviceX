struct Quants {
    int n, nc, nt, nv;
    Particle *pp, *pp_hst; /* vertices */
    int *adj0, *adj1;      /* adjacency lists */
    int4 *tri;             /* triangles */

    int *tri_hst;
    float *av;
};

/* textures ticket */
struct TicketT {
    Texo <float2> texvert;
    Texo <int> texadj0, texadj1;
    Texo <int4> textri;
};


void alloc_quants(Quants *q) {
    q->n = q->nc = 0;
    mpDeviceMalloc(&q->pp);
    q->pp_hst = new Particle[MAX_PART_NUM];

    q->nt = RBCnt;
    q->nv = RBCnv;

    CC(cudaMalloc(&q->tri, q->nt * sizeof(int4)));
    CC(cudaMalloc(&q->adj0, q->nv * RBCmd * sizeof(int)));
    CC(cudaMalloc(&q->adj1, q->nv * RBCmd * sizeof(int)));
                      
    q->tri_hst = new int[MAX_FACE_NUM];
    CC(cudaMalloc(&q->av, 2*MAX_CELL_NUM));
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp));
    CC(cudaFree(q->av));

    CC(cudaFree(q->tri));
    CC(cudaFree(q->adj0));
    CC(cudaFree(q->adj1));

    q->textri.destroy();
    q->texadj0.destroy();
    q->texadj1.destroy();
    q->texvert.destroy();

    delete[] q->tri_hst;
    delete[] q->pp_hst;
}

void setup(Quants *q) {
    sub::setup(q->tri_hst, q->tri, &q->textri, q->adj0, &q->texadj0, q->adj1, &q->texadj1, q->pp, &q->texvert, q->n);
}

void setup_textures(const Quants *q, TicketT *t) {
    sub::setup(q->tri, &t->textri, q->adj0, &t->texadj0, q->adj1, &t->texadj1, q->pp, &t->texvert);
}

void forces(const Quants q, /**/ Force *ff) {
    sub::forces(q.nc, q.texvert, q.textri, q.texadj0, q.texadj1, /**/ ff, q.av);
}

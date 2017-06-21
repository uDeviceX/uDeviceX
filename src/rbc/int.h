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

    delete[] q->tri_hst;
    delete[] q->pp_hst;
}

void setup(const char *r_templ, Quants *q) {
    sub::setup(r_templ, /**/ q->tri_hst, q->tri, q->adj0, q->adj1);
}

void gen_quants(const char *r_templ, const char *r_state, Quants *q) {
    ic::setup_from_pos(r_templ, r_state, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
}

void strt_quants(const int id, Quants *q) {
    sub::setup_from_strt(id, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
}

void gen_ticket(const Quants q, TicketT *t) {
    sub::setup_textures(q.tri, &t->textri, q.adj0, &t->texadj0, q.adj1, &t->texadj1, q.pp, &t->texvert);
}

void destroy_textures(TicketT *t) {
    t->textri.destroy();
    t->texadj0.destroy();
    t->texadj1.destroy();
    t->texvert.destroy();
}

void forces(const Quants q, const TicketT t, /**/ Force *ff) {
    sub::forces(q.nc, t.texvert, t.textri, t.texadj0, t.texadj1, /**/ ff, q.av);
}

void strt_dump(const int id, const Quants q) {
    sub::strt_dump(id, q.n, q.pp, /*w*/ q.pp_hst);
}

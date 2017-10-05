void ini(Quants *q) {
    q->n = q->nc = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    q->pp_hst = (Particle*) malloc(MAX_PART_NUM * sizeof(Particle));

    q->nt = RBCnt;
    q->nv = RBCnv;

    Dalloc(&q->tri,  q->nt);
    Dalloc(&q->adj0, q->nv * RBCmd);
    Dalloc(&q->adj1, q->nv * RBCmd);

    q->tri_hst = (int4*) malloc(MAX_FACE_NUM * sizeof(int4));
    Dalloc(&q->av, 2*MAX_CELL_NUM);

    if (rbc_ids)
        q->ii = (int*) malloc(MAX_CELL_NUM * sizeof(int));
}

static void setup_textures(int md, int nt, int nv, int4 *tri, Texo<int4> *textri, int *adj0, Texo<int> *texadj0,
                           int *adj1, Texo<int> *texadj1, Particle *pp, Texo<float2> *texvert) {
    TE(texadj0, adj0, nv*md);
    TE(texadj1, adj1, nv*md);
    TE(textri,  tri,  nt);
    TE(texvert, (float2*)pp, 3*MAX_PART_NUM);
}

void gen_ticket(const Quants q, TicketT *t) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    setup_textures(md, nt, nv, q.tri, &t->textri, q.adj0, &t->texadj0, q.adj1, &t->texadj1, q.pp, &t->texvert);
}

void ini(int maxcells, /**/ ComHelper *com) {
    size_t sz = maxcells * sizeof(float3);
    CC(d::alloc_pinned((void**) &com->hrr, sz));
    CC(d::Malloc((void**) &com->drr, sz));
}

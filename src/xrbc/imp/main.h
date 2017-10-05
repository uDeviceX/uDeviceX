static void forces(int nc, const Texo<float2> texvert, const Texo<int4> textri, const Texo<int> texadj0, const Texo<int> texadj1, Force *ff, float* av) {
    if (nc <= 0) return;

    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;

    Dzero(av, 2*nc);
    KL(dev::area_volume, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, texvert, texadj0, texadj1, nc, av, (float*)ff));
}

void forces(const Quants q, const TicketT t, /**/ Force *ff) {
    forces(q.nc, t.texvert, t.textri, t.texadj0, t.texadj1, /**/ ff, q.av);
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

void destroy_textures(TicketT *t) {
    t->textri.destroy();
    t->texadj0.destroy();
    t->texadj1.destroy();
    t->texvert.destroy();
}


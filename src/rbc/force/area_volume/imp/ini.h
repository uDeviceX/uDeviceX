static void setup_textures(int md, int nt, int nv, int4 *tri, Texo<int4> *textri, int *adj0, Texo<int> *texadj0,
                           int *adj1, Texo<int> *texadj1, Particle *pp, Texo<float2> *texvert) {
    TE(texadj0, adj0, nv*md);
    TE(texadj1, adj1, nv*md);
    TE(textri,  tri,  nt);
    TE(texvert, (float2*)pp, 2*MAX_PART_NUM);
}

void gen_ticket(const Quants q, TicketT *t) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    setup_textures(md, nt, nv, q.tri, &t->textri, q.adj0, &t->texadj0, q.adj1, &t->texadj1, q.pp, &t->texvert);
}

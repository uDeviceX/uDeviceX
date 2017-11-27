static void setup_textures(Particle *pp, Texo<float2> *texvert) {
    TE(texvert, (float2*)pp, 2*MAX_PART_NUM);
}

static void setup_rnd(int md, int nv, rbc::rnd::D **prnd) {
    int n;
    long seed;
    n = nv*md*MAX_CELL_NUM;
    seed = rbc::rnd::ENV;
    rbc::rnd::ini(prnd, n, seed);
}

void gen_ticket(const Quants q, TicketT *t) {
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    setup_textures(q.pp, &t->texvert);
    if (RBC_RND) setup_rnd(md, nv, &t->rnd);
}

static void setup_rnd(int md, int nv, rbc::rnd::D **prnd) {
    int n;
    long seed;
    n = nv*md*MAX_CELL_NUM;
    seed = rbc::rnd::ENV;
    rbc::rnd::ini(prnd, n, seed);
}

void gen_ticket(const RbcQuants q, RbcForce *t) {
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    if (RBC_RND) setup_rnd(md, nv, &t->rnd);
}

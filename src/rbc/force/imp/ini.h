static void setup_rnd(int md, int nv, RbcRnd **prnd) {
    int n;
    long seed;
    n = nv*md*MAX_CELL_NUM;
    seed = ENV;
    rbc_rnd_ini(prnd, n, seed);
}

void rbc_force_gen(const RbcQuants q, RbcForce *t) {
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    if (RBC_RND) setup_rnd(md, nv, &t->rnd);
}

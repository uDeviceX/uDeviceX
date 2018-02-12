static void setup_rnd(int md, int nv, RbcRnd **prnd) {
    int n;
    long seed;
    n = nv*md*MAX_CELL_NUM;
    seed = ENV;
    rbc_rnd_ini(prnd, n, seed);
}

void rbc_force_ini(const RbcQuants *d, RbcForce **pq) {
    RbcForce *q;
    int md, nv;
    EMALLOC(1, &q);
    md = RBCmd;
    nv = d->nv;
    if (nv <= 0) ERR("nv=%d < 0", nv);
    if (RBC_RND) setup_rnd(md, nv, &q->rnd);
    *pq = q;
}

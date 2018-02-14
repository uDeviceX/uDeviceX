static void setup_rnd(int md, int nv, RbcRnd **prnd) {
    int n;
    long seed;
    n = nv*md*MAX_CELL_NUM;
    seed = ENV;
    rbc_rnd_ini(prnd, n, seed);
}

void rbc_force_ini(int nv, RbcForce **pq) {
    RbcForce *q;
    int md;
    if (nv <= 0) ERR("nv=%d < 0", nv);    
    EMALLOC(1, &q);
    md = RBCmd;
    if (RBC_RND) setup_rnd(md, nv, &q->rnd);
    *pq = q;
}

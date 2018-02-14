void rbc_force_ini(int nv, int seed, RbcForce **pq) {
    RbcForce *q;
    int md;
    if (nv <= 0) ERR("nv=%d < 0", nv);    
    EMALLOC(1, &q);
    md = RBCmd;
    if (RBC_RND) rbc_rnd_ini(nv*md*MAX_CELL_NUM, seed, &q->rnd);
    *pq = q;
}

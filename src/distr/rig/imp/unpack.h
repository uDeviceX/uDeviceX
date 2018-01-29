void drig_unpack_bulk(const DRigPack *p, /**/ RigQuants *q) {
    int ns, nv, n;
    ns = p->hipp.counts[frag_bulk];
    nv = q->nv;
    n = ns * nv;
    
    if (ns) {
        CC(d::MemcpyAsync(q->i_pp, p->dipp.data[frag_bulk], n * sizeof(Particle), D2D));
        CC(d::MemcpyAsync(  q->ss,  p->dss.data[frag_bulk],   ns * sizeof(Solid), D2D));
    }
    q->ns = ns;
}

static void shift(int fid, float r[3]) {
    enum {X, Y, Z};
    r[X] += XS * frag_i2d(fid, X);
    r[Y] += YS * frag_i2d(fid, Y);
    r[Z] += ZS * frag_i2d(fid, Z);
}

static void shift_ss_one_frag(int n, int fid, Solid *ss) {
    for (int i = 0; i < n; ++i) shift(fid, ss[i].com);
}

void drig_unpack_halo(const DRigUnpack *u, /**/ RigQuants *q) {
    int ns, nv, n, i, strtp, strts;
    size_t szp, szs;
    nv = q->nv;
    strts = q->ns;
    strtp = strts * nv;
    
    for (i = 0; i < NFRAGS; ++i) {
        ns = u->hss.counts[i];
        n  = ns * nv; 
        szp = n * sizeof(Particle);
        szs = ns * sizeof(Solid);

        if (ns) {
            /* particles */
            CC(d::MemcpyAsync(q->i_pp + strtp, u->hipp.data[i], szp, H2D));
            dcommon_shift_one_frag(u->L, n, i, /**/ q->i_pp + strtp);
            /* solid */
            shift_ss_one_frag(ns, i, /**/ (Solid*) u->hss.data[i]);
            CC(d::MemcpyAsync(q->ss + strts, u->hss.data[i], szs, H2D));
        }
        strtp += n;
        strts += ns;
    }
    q->ns = strts;
}

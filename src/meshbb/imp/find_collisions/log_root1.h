static void log_cubic_reset() {
    int n = 0;
    CC(d::MemcpyToSymbol(&dev::ncubicInfo, &n, sizeof(n)));
}

static void log_cubic_dump() {
    dev::CubicInfo cci[MAX_CUBIC_INFO];
}

void find_collisions(int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                     const int *starts, const int *counts, const Particle *pp, const Force *ff,
                     /**/ BBdata d) {
    log_cubic_reset();    
    if (!nm) return;
    KL(dev::find_collisions, (k_cnf(nm * nt)),
       (nm, nt, nv, tt, i_pp, L, starts, counts, pp, ff, /**/ d.ncols, d.datacol, d.idcol));
    log_cubic_dump();
}

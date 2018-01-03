static void log_cubic_reset() {
    int n = 0;
    CC(d::MemcpyToSymbol(&dev::ncubicInfo, &n, sizeof(n)));
}

static void log_cubic_dump0(int n, dev::CubicInfo *cci) {
    int i;
    dev::CubicInfo ci;
    for (i = 0; i < n; i++) {
        ci = cci[i];
        msg_print("%.16e %.16e %.16e %.16e %.16e %d :log_root:", ci.a, ci.b, ci.c, ci.d, ci.h, ci.status);
    }
}
static void log_cubic_dump() {
    int n;
    dev::CubicInfo cci[MAX_CUBIC_INFO];
    CC(d::MemcpyFromSymbol(&n,   &dev::ncubicInfo,   sizeof(n)));
    CC(d::MemcpyFromSymbol(&cci, &dev::cubicInfo,  n*sizeof(cci[0])));
    log_cubic_dump0(n, cci);
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

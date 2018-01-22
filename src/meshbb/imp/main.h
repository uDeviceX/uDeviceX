void meshbb_ini(int maxpp, /**/ MeshBB *d) {
    CC(d::Malloc((void**) &d->ncols,   maxpp * sizeof(int)));
    CC(d::Malloc((void**) &d->datacol, maxpp * MAX_COL * sizeof(float4)));
    CC(d::Malloc((void**) &d->idcol,   maxpp * MAX_COL * sizeof(int)));
}

void meshbb_fin(/**/ MeshBB *d) {
    CC(d::Free(d->ncols));
    CC(d::Free(d->datacol));
    CC(d::Free(d->idcol));    
}

void meshbb_reini(int n, /**/ MeshBB *d) {
    CC(d::MemsetAsync(d->ncols, 0, n * sizeof(int)));
}

void meshbb_select_collisions(int n, /**/ MeshBB *d) {
    KL(dev::select_collisions, (k_cnf(n)), (n, /**/ d->ncols, d->datacol, d->idcol));
}

void meshbb_bounce(int n, const MeshBB *d, const Force *ff, int nt, int nv, const int4 *tt, const Particle *i_pp,
            /**/ Particle *pp, Momentum *mm) {
    KL(dev::perform_collisions, (k_cnf(n)),
       (n, d->ncols, d->datacol, d->idcol, ff, nt, nv, tt, i_pp, /**/ pp, mm));
}

void meshbb_collect_rig_momentum(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
    KL(dev::collect_rig_mom, (k_cnf(ns * nt)), (ns, nt, nv, tt, pp, mm, /**/ ss));
}

void meshbb_collect_rbc_momentum(int nc, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Force *ff) {
    KL(dev::collect_rbc_mom, (k_cnf(nc * nt)), (nc, nt, nv, tt, pp, mm, /**/ ff));
}

void meshbb_ini(int maxpp, /**/ MeshBB **meshbb) {
    MeshBB *mbb;
    EMALLOC(1, meshbb);
    mbb = *meshbb;
    Dalloc(&mbb->ncols,   maxpp);
    Dalloc(&mbb->datacol, maxpp * MAX_COL);
    Dalloc(&mbb->idcol,   maxpp * MAX_COL);
}

void meshbb_fin(/**/ MeshBB *mbb) {
    Dfree(mbb->ncols);
    Dfree(mbb->datacol);
    Dfree(mbb->idcol);
    EFREE(mbb);
}

void meshbb_reini(int n, /**/ MeshBB *mbb) {
    CC(d::MemsetAsync(mbb->ncols, 0, n * sizeof(int)));
}

void meshbb_select_collisions(int n, /**/ MeshBB *mbb) {
    KL(meshbb_dev::select_collisions, (k_cnf(n)), (n, /**/ mbb->ncols, mbb->datacol, mbb->idcol));
}

void meshbb_bounce(float dt, float mass,
                   int n, const MeshBB *mbb, const Force *ff, MeshInfo mi, const Particle *i_pp,
                   /**/ Particle *pp, Momentum *mm) {
    KL(meshbb_dev::perform_collisions, (k_cnf(n)),
       (dt, mass, n, mbb->ncols, mbb->datacol, mbb->idcol, ff, mi.nt, mi.nv, mi.tt, i_pp, /**/ pp, mm));
}

void meshbb_collect_rig_momentum(float dt, int ns, MeshInfo mi, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
    KL(meshbb_dev::collect_rig_mom, (k_cnf(ns * mi.nt)), (dt, ns, mi.nt, mi.nv, mi.tt, pp, mm, /**/ ss));
}

void meshbb_collect_rbc_momentum(float dt, int nc, MeshInfo mi, const Particle *pp, const Momentum *mm, /**/ Force *ff) {
    KL(meshbb_dev::collect_rbc_mom, (k_cnf(nc * mi.nt)), (dt, nc, mi.nt, mi.nv, mi.tt, pp, mm, /**/ ff));
}

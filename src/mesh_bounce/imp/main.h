void mesh_bounce_ini(int maxpp, /**/ MeshBB **meshbb) {
    MeshBB *mbb;
    EMALLOC(1, meshbb);
    mbb = *meshbb;
    Dalloc(&mbb->ncols,   maxpp);
    Dalloc(&mbb->datacol, maxpp * MAX_COL);
    Dalloc(&mbb->idcol,   maxpp * MAX_COL);
}

void mesh_bounce_fin(/**/ MeshBB *mbb) {
    Dfree(mbb->ncols);
    Dfree(mbb->datacol);
    Dfree(mbb->idcol);
    EFREE(mbb);
}

void mesh_bounce_reini(int n, /**/ MeshBB *mbb) {
    CC(d::MemsetAsync(mbb->ncols, 0, n * sizeof(int)));
}

void mesh_bounce_find_collisions(float dt, int nm, MeshInfo mi, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0,
                                 /**/ MeshBB *d) {
    if (!nm) return;
    KL(mesh_bounce_dev::find_collisions, (k_cnf(nm * mi.nt)),
       (dt, nm, mi.nt, mi.nv, mi.tt, i_rr, L, starts, counts, pp, pp0, /**/ d->ncols, d->datacol, d->idcol));
}

void mesh_bounce_select_collisions(int n, /**/ MeshBB *mbb) {
    KL(mesh_bounce_dev::select_collisions, (k_cnf(n)), (n, /**/ mbb->ncols, mbb->datacol, mbb->idcol));
}

void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *mbb, MeshInfo mi, const Positioncp *i_rr,
                   const Particle *pp0, /**/ Particle *pp, Momentum *mm) {
    KL(mesh_bounce_dev::perform_collisions, (k_cnf(n)),
       (dt, mass, n, mbb->ncols, mbb->datacol, mbb->idcol, mi.nt, mi.nv, mi.tt, i_rr, pp0, /**/ pp, mm));
}

void mesh_bounce_collect_rig_momentum(int ns, MeshInfo mi, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
    KL(mesh_bounce_dev::collect_rig_mom, (k_cnf(ns * mi.nt)), (ns, mi.nt, mi.nv, mi.tt, pp, mm, /**/ ss));
}

void mesh_bounce_collect_rbc_momentum(float dt, int nc, MeshInfo mi, const Particle *pp, const Momentum *mm, /**/ Force *ff) {
    KL(mesh_bounce_dev::collect_rbc_mom, (k_cnf(nc * mi.nt)), (dt, nc, mi.nt, mi.nv, mi.tt, pp, mm, /**/ ff));
}

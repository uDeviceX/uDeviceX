void meshbb_find_collisions(float dt,
                            int nm, MeshInfo mi, const Particle *i_pp, int3 L,
                            const int *starts, const int *counts, const Particle *pp, const Force *ff,
                            /**/ MeshBB *d) {
    if (!nm) return;
    KL(meshbb_dev::find_collisions, (k_cnf(nm * mi.nt)),
       (dt, nm, mi.nt, mi.nv, mi.tt, i_pp, L, starts, counts, pp, ff, /**/ d->ncols, d->datacol, d->idcol));
}

void meshbb_find_collisions(float dt,
                            int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                            const int *starts, const int *counts, const Particle *pp, const Force *ff,
                            /**/ MeshBB *d) {
    if (!nm) return;
    KL(dev::find_collisions, (k_cnf(nm * nt)),
       (dt, nm, nt, nv, tt, i_pp, L, starts, counts, pp, ff, /**/ d->ncols, d->datacol, d->idcol));
}

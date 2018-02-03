struct MeshBB;

void meshbb_ini(int maxpp, /**/ MeshBB **d);
void meshbb_fin(/**/ MeshBB *d);

void meshbb_reini(int n, /**/ MeshBB *d);
void meshbb_find_collisions(float dt,
                            int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                            const int *starts, const int *counts, const Particle *pp, const Force *ff, /**/ MeshBB *d);
void meshbb_select_collisions(float dt,
                              int n, /**/ MeshBB *d);
void meshbb_bounce(float dt,
                   int n, const MeshBB *d, const Force *ff, int nt, int nv, const int4 *tt,
                   const Particle *i_pp, /**/ Particle *pp, Momentum *mm);
void meshbb_collect_rig_momentum(float dt,
                                 int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss);
void meshbb_collect_rbc_momentum(float dt,
                                 int nc, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Force *ff);

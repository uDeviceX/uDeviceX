namespace meshbb {

struct MeshBB {
    int *ncols;       /* number of possible collisions per particle      */
    float4 *datacol;  /* list of data related to collisions per particle */
    int *idcol;       /* list of triangle colliding ids per particle     */
};

void meshbb_ini(int maxpp, /**/ MeshBB *d);
void meshbb_fin(/**/ MeshBB *d);

void meshbb_reini(int n, /**/ MeshBB d);
void meshbb_find_collisions(int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                     const int *starts, const int *counts, const Particle *pp, const Force *ff, /**/ MeshBB d);
void meshbb_select_collisions(int n, /**/ MeshBB d);
void meshbb_bounce(int n, MeshBB d, const Force *ff, int nt, int nv, const int4 *tt, const Particle *i_pp, /**/ Particle *pp, Momentum *mm);

void meshbb_collect_rig_momentum(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss);
void meshbb_collect_rbc_momentum(int nc, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Force *ff);

} // meshbb

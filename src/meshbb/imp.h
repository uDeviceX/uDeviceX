struct MeshBB;
struct Particle;
struct Force;
struct Momentum;
struct Solid;
struct int4;

struct MeshInfo {
    int nt, nv;
    const int4 *tt;
};

// tag::mem[]
void meshbb_ini(int maxpp, /**/ MeshBB **d);
void meshbb_fin(/**/ MeshBB *d);
// end::mem[]

// tag::int[]
void meshbb_reini(int n, /**/ MeshBB *d);  // <1>
void meshbb_find_collisions(float dt, int nm, MeshInfo meshinfo, const Particle *i_pp, int3 L,
                            const int *starts, const int *counts, const Particle *pp, const Force *ff, /**/ MeshBB *d);  // <2>
void meshbb_select_collisions(int n, /**/ MeshBB *d);  // <3>
void meshbb_bounce(float dt, float mass, int n, const MeshBB *d, const Force *ff, MeshInfo meshinfo,
                   const Particle *i_pp, /**/ Particle *pp, Momentum *mm);  // <4>
// end::int[]

// tag::collect[]
void meshbb_collect_rig_momentum(float dt, int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Solid *ss); // <1>
void meshbb_collect_rbc_momentum(float dt, int nc, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Force *ff); // <2>
// end::collect[]

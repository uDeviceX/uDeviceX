struct MeshBB;

struct Particle;
struct Positioncp;
struct Force;
struct Momentum;
struct Rigid;
struct int4;

struct MeshInfo {
    int nt, nv;
    const int4 *tt;
};

// tag::mem[]
void mesh_bounce_ini(int maxpp, /**/ MeshBB **d);
void mesh_bounce_fin(/**/ MeshBB *d);
// end::mem[]

// tag::int[]
void mesh_bounce_reini(int n, /**/ MeshBB *d);  // <1>
void mesh_bounce_find_collisions(float dt, int nm, MeshInfo meshinfo, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0, /**/ MeshBB *d);  // <2>
void mesh_bounce_select_collisions(int n, /**/ MeshBB *d);  // <3>
void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *d, MeshInfo meshinfo,
                        const Positioncp *i_rr, const Particle *pp0, /**/ Particle *pp, Momentum *mm);  // <4>
// end::int[]

// tag::collect[]
void mesh_bounce_collect_rig_momentum(int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Rigid *ss); // <1>
void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm, /**/ Particle *pp);            // <2>
// end::collect[]

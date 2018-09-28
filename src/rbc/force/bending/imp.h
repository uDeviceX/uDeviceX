struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;
struct RbcParams;

// tag::mem[]
void rbc_bending_ini(const MeshRead *cell, RbcForce**);
void rbc_bending_fin(RbcForce*);
// end::mem[]

// tag::apply[]
void rbc_bending_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]

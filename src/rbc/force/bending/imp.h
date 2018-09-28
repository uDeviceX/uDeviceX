struct RbcParams;
struct RbcBending;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void rbc_bending_ini(const MeshRead *cell, RbcBending**);
void rbc_bending_fin(RbcBending*);
// end::mem[]

// tag::apply[]
void rbc_bending_apply(RbcBending*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]

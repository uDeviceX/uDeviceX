struct RbcParams;
struct Juelicher;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void juelicher_ini(const MeshRead*, Juelicher**);
void juelicher_fin(Juelicher*);
// end::mem[]

// tag::apply[]
void juelicher_apply(Juelicher*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]

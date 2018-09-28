struct RbcParams;
struct Kantor;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void kantor_ini(const MeshRead *cell, Kantor**);
void kantor_fin(Kantor*);
// end::mem[]

// tag::apply[]
void kantor_apply(Kantor*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]

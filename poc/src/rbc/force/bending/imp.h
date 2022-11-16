struct RbcParams;
struct Bending;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void bending_kantor_ini(const MeshRead*, Bending**);
void bending_juelicher_ini(const MeshRead*, Bending**);
void bending_none_ini(const MeshRead*, Bending**);

void bending_fin(Bending*);
// end::mem[]

// tag::apply[]
void bending_apply(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]

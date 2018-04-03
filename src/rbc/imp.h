struct Coords;
struct MeshRead;

// tag::mem[]
void rbc_ini(bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);
// end::mem[]

// tag::ini[]
void rbc_gen_quants(const Coords*, MPI_Comm, MeshRead*, const char *ic, RbcQuants*); // <1>
void rbc_strt_quants(const Coords*, MeshRead*, const int id, RbcQuants*); // <2>
void rbc_strt_dump(const Coords*, int id, const RbcQuants *q); // <3>
// end::ini[]


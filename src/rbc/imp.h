struct Coords;
struct MeshRead;

// tag::mem[]
void rbc_ini(long maxnc, bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);
// end::mem[]

// tag::ini[]
void rbc_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RbcQuants*); // <1>
void rbc_gen_freeze(MPI_Comm, /**/ RbcQuants*);                                               // <2>
void rbc_strt_quants(MPI_Comm, const MeshRead*, const char *base, const int id, RbcQuants*);  // <3>
void rbc_strt_dump(MPI_Comm, const char *base, const char *name, int id, const RbcQuants *q); // <4>
// end::ini[]


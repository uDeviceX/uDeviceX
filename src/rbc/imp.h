struct Coords;
struct MeshRead;

void rbc_ini(bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);

void rbc_gen_quants(const Coords*, MPI_Comm comm, MeshRead*, const char *ic, RbcQuants*);
void rbc_strt_quants(const Coords*, MeshRead*, const int id, RbcQuants*);
void rbc_strt_dump(const Coords*, int id, const RbcQuants *q);

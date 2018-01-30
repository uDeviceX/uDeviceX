struct Coords;
struct OffRead;

void rbc_ini(RbcQuants*);
void rbc_fin(RbcQuants*);

void rbc_gen_quants(const Coords*, MPI_Comm comm, const char *cell, OffRead*, const char *ic, RbcQuants*);
void rbc_strt_quants(const Coords*, OffRead*, const int id, RbcQuants*);
void rbc_strt_dump(const Coords*, const int id, const RbcQuants *q);

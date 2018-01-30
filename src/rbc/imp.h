struct Coords;
struct OffRead;

void rbc_ini(RbcQuants *q);
void rbc_fin(RbcQuants *q);

void rbc_gen_quants(const Coords *coords, MPI_Comm comm, const char *cell, OffRead *off, const char *ic, RbcQuants *q);
void rbc_strt_quants(const Coords *coords, const char *cell, const int id, RbcQuants *q);
void rbc_strt_dump(const Coords *coords, const int id, const RbcQuants *q);

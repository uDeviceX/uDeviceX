namespace rbc { namespace main {

void ini(RbcQuants *q);
void fin(RbcQuants *q);

void gen_quants(Coords coords, MPI_Comm comm, const char *cell, const char *ic, RbcQuants *q);
void strt_quants(Coords coords, const char *cell, const int id, RbcQuants *q);
void strt_dump(Coords coords, const int id, const RbcQuants *q);

}} /* namespace */

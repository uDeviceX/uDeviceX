namespace rbc { namespace main {

void ini(Quants *q);
void fin(Quants *q);

void gen_quants(MPI_Comm comm, const char *cell, const char *ic, Quants *q);
void strt_quants(Coords coords, const char *cell, const int id, Quants *q);
void strt_dump(Coords coords, const int id, const Quants *q);

}} /* namespace */

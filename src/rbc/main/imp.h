namespace rbc { namespace main {

void ini(Quants *q);
void fin(Quants *q);

void gen_quants(const char *cell, const char *r_state, Quants *q);
void strt_quants(const char *cell, const int id, Quants *q);
void strt_dump(const int id, const Quants q);

}} /* namespace */

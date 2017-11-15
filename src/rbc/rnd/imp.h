namespace rbc { namespace rnd {
struct D;
void ini(D**, int n, int seed);
void fin(D*);
void gen(D*, int n);
float get_hst(const D*, int i);
}} /* namespace */

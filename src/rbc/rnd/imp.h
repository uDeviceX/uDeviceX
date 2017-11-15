namespace rbc { namespace rnd {
struct D;
enum {ENV = -2, TIME = -1}; /* special seeds */
void ini(D**, int n, long seed);
void fin(D*);
void gen(D*, int n);
float get_hst(const D*, int i);
}} /* namespace */

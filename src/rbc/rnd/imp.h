namespace rbc { namespace rnd {
struct RbcRnd;
enum {ENV = -2, TIME = -1}; /* special seeds */
void ini(RbcRnd**, int n, long seed);
void fin(RbcRnd*);
void gen(RbcRnd*, int n);
float get_hst(const RbcRnd*, int i);
}} /* namespace */

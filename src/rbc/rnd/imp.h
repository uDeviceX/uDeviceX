struct RbcRnd;
enum {ENV = -2, TIME = -1}; /* special seeds */
void rbc_rnd_ini(RbcRnd**, int n, long seed);
void rbc_rnd_fin(RbcRnd*);
void rbc_rnd_gen(RbcRnd*, int n);
float rbc_rnd_get_hst(const RbcRnd*, int i);

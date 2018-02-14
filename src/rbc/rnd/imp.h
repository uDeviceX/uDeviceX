struct RbcRnd;
enum {ENV = -2, TIME = -1}; /* special seeds */
void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);
void rbc_rnd_gen(RbcRnd*, int n, /**/ float**);
float rbc_rnd_get_hst(const RbcRnd*, int i);

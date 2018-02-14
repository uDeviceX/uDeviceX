struct RbcRnd;
enum {SEED_TIME = -1}; /* magic seed: ini from "time" */
void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);
void rbc_rnd_gen(RbcRnd*, int n, /**/ float**);
float rbc_rnd_get_hst(const RbcRnd*, int i);

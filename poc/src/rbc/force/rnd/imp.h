struct RbcRnd;

// tag::enum[]
enum {SEED_TIME = -1}; /* magic seed: ini from "time" */
// end::enum[]

// tag::mem[]
void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);
// end::mem[]

// tag::int[]
void rbc_rnd_gen(RbcRnd*, int n, /**/ float**); // <1>
float rbc_rnd_get_hst(const RbcRnd*, int i);    // <2>
// end::int[]

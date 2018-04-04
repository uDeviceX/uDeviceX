struct int3;
struct Particle;
// tag::int[]
void wall_exch_pp(MPI_Comm cart, int3 L, int maxn, /*io*/ Particle *pp, int *n);
// end::int[]

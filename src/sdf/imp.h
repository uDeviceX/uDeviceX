namespace sdf {
struct Quants;
void alloc_quants(Quants**);
void  free_quants(Quants*);
void ini(MPI_Comm cart, Quants*);
void bulk_wall(const Quants*, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(const Quants*, Particle *pp, int n, int nc, int nv, int *stay);
void bounce(const Quants*, int n, /**/ Particle*);
}

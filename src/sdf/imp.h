struct Sdf;
void alloc_quants(Sdf**);
void  free_quants(Sdf*);

void gen(MPI_Comm cart, Sdf*);

void bulk_wall(const Sdf*, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(const Sdf*, Particle *pp, int n, int nc, int nv, int *stay);
void bounce(const Sdf*, int n, /**/ Particle*);

struct Sdf;
void ini(Sdf**);
void fin(Sdf*);

void gen(MPI_Comm cart, Sdf*);

void bulk_wall(const Sdf*, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(const Sdf*, Particle *pp, int n, int nc, int nv, int *stay);
void bounce(Wvel_d wv, Coords c, const Sdf*, int n, /**/ Particle*);

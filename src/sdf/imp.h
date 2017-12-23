struct Sdf;
struct Sdf_v;

void ini(Sdf**);
void fin(Sdf*);
void gen(MPI_Comm cart, Sdf*);
void to_view(Sdf*, /**/ Sdf_v*);

void bulk_wall(const Sdf*, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(const Sdf*, Particle *pp, int n, int nc, int nv, int *stay);
void bounce(Wvel_v wv, Coords c, const Sdf*, int n, /**/ Particle*);

struct Sdf;
struct Sdf_v;
struct Wvel_v;
struct Coords;
struct Particle;

void sdf_ini(Sdf**);
void sdf_fin(Sdf*);
void gen(Coords*, MPI_Comm cart, Sdf*);
void to_view(Sdf*, /**/ Sdf_v*);

void bulk_wall(Sdf*, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(Sdf*, Particle*, int n, int nc, int nv, /**/ int *stay);
void bounce(Wvel_v*, Coords*, Sdf*, int n, /**/ Particle*);

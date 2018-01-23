struct Sdf;
struct Sdf_v;
struct Wvel_v;
struct Coords;
struct Particle;

void sdf_ini(Sdf**);
void sdf_fin(Sdf*);
void sdf_gen(const Coords*, MPI_Comm cart, Sdf*);
void sdf_to_view(const Sdf*, /**/ Sdf_v*);

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp);
int  sdf_who_stays(const Sdf*, const Particle*, int n, int nc, int nv, /**/ int *stay);
void sdf_bounce(const Wvel_v*, const Coords*, const Sdf*, int n, /**/ Particle*);

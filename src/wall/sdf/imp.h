struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);
void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*);
void sdf_to_view(const Sdf*, /**/ Sdf_v*);

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp);
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);
void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);

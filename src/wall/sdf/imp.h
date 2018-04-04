struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

// tag::mem[]
void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);
// end::mem[]

// tag::int[]
void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_to_view(const Sdf*, /**/ Sdf_v*);                   // <2>
// end::int[]

// tag::tools[]
void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>
// end::tools[]

// tag::bounce[]
void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);
// end::bounce[]

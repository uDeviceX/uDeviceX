struct Wall;

struct OptParams;
struct Config;
struct PairParams;
struct Coords;

// tag::mem[]
void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);
// end::mem[]

// tag::gen[]
void wall_gen(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
              /*io*/ int *n, Particle *pp, /**/ Wall*);
// end::gen[]

// tag::strt[]
void wall_restart(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                  const char *base, /**/ Wall*);
// end::strt[]

// tag::dump[]
void wall_dump_templ(const Wall*, MPI_Comm, const char *base);
// end::dump[]

// tag::main[]
void wall_interact(const Coords*, const PairParams*, Wall*, PFarrays*);
void wall_adhesion(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_bounce(const Wall*, const Coords*, float dt, PFarrays*);
void wall_repulse(const Wall*, PFarrays*);
// end::main[]

// tag::upd[]
void wall_update_vel(float t, Wall *);
// end::upd[]

// tag::get[]
void wall_get_sdf_ptr(const Wall*, const Sdf**);           // <1>
double wall_compute_volume(const Wall*, MPI_Comm, int3 L); // <2>
// end::get[]

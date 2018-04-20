struct Wall;

struct OptParams;
struct Config;
struct PairParams;
struct Coords;

void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);

void wall_gen(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
              /*io*/ int *n, Particle *pp, /**/ Wall*);

void wall_restart(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                  const char *base, /**/ Wall*);

int  wall_who_stays(const Wall*, int n, const Particle*, int nc, int nv, /**/ int *stay);

void wall_interact(const Coords*, const PairParams*, Wall*, PFarrays*);

/* maximum particle number per one processor for static allocation */
#define MAX_PART_NUM 1000000

/* maximum number of particles per solid */
#define MAX_PSOLID_NUM 30000

/* maximum number of solids per node */
#define MAX_SOLIDS 20

/* maximum number of faces per one RBC */
#define MAX_FACE_NUM 5000
#define MAX_VERT_NUM 10000
#define MAX_CELL_NUM 10000

/* safety factor for dpd halo interactions */
#define HSAFETY_FACTOR 10.f

/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII

#define MSG00(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define MSG(fmt, ...) MSG00("%03d: ", m::rank), MSG00(fmt, ##__VA_ARGS__), MSG00("\n")
#define MSG0(fmt, ...) do { if (m::rank == 0) MSG(fmt, ##__VA_ARGS__); } while (0)

#define ERR(fmt, ...) do { fprintf(stderr, "%03d: ERROR: %s, l%d: " fmt, m::rank, __FILE__, __LINE__, ##__VA_ARGS__); exit(1); } while(0)

template <typename T, int N>
struct Sarray { /* [s]tatic array */
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Solid {
    float Iinv[6],            /* moment of inertia            6        */
        mass,                 /* mass of the solid            7        */
        com[3],               /* [c]enter [o]f [m]ass         10       */
        v[3], om[3],          /* linear and angular velocity  13 16    */
        e0[3], e1[3], e2[3],  /* local referential            19 22 25 */
        fo[3], to[3],         /* force, torque                28 31    */
        id;                   /* id of the solid              32       */
};

struct Mesh {   /* triangle mesh structure                */
    int nv, nt; /* number of [v]ertices and [t]riangles   */
    int *tt;    /* triangle indices t1 t2 t3 t1 t2 t3 ... */
    float *vv;  /* vertices x y z x y z ...               */
};

struct Force {
    float f[3];
};

struct ParticlesWrap {
    const Particle *p;
    Force *f;
    int n;
    ParticlesWrap() : p(NULL), f(NULL), n(0) {}
    ParticlesWrap(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};

struct SolventWrap : ParticlesWrap {
    const int *cellsstart, *cellscount;
    SolventWrap() : cellsstart(NULL), cellscount(NULL), ParticlesWrap() {}
    SolventWrap(const Particle *const p, const int n, Force *f,
                const int *const cellsstart, const int *const cellscount)
        : ParticlesWrap(p, n, f),
          cellsstart(cellsstart),
          cellscount(cellscount) {}
};

void diagnostics(Particle *_particles, int n, int idstep);

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

struct ParticlesWrap000 {
    const Particle *p;
    int n;
    Force *f;
    ParticlesWrap000() : n(0) {}
    ParticlesWrap000(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};

struct SolventWrap : ParticlesWrap000 {
    const int   *cellscount, *cellsstart;
    SolventWrap() : ParticlesWrap000() {}
    SolventWrap(const Particle *const p, const int n, Force *f,
                const int *const cellsstart, const int *const cellscount)
        : ParticlesWrap000(p, n, f),
          cellscount(cellscount),
          cellsstart(cellsstart) {
    }
};

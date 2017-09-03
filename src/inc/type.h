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

struct SolventWrap {
    Particle *pp;
    Force *ff;
    int n; 
    int *starts;
};

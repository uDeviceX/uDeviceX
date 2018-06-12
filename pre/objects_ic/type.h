typedef float real;

struct Particles {
    real *xx, *yy, *zz;
    real *vx, *vy, *vz;
    real *fx, *fy, *fz;
};

enum {
    ANGLE_RND,
    ANGLE_IN
};

struct Angle {
    int type;
    /* angles: rotate around x, y then z */
    real x, y, z;
};

struct Args {
    int n;
    int Lx, Ly, Lz;
    real r, sc;
    Angle ang;
    bool dump_xyz;
};

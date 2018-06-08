typedef float real;

struct Args {
    int n;
    int Lx, Ly, Lz;
    real r;
};

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

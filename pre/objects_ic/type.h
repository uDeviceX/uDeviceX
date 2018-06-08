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

enum {
    RHO, VX, VY, VZ,
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS
};

static const char *names[NFIELDS] = {
    "density", "u", "v", "w",
    "sxx", "sxy", "sxz",
    "syy", "syz", "szz"
};

struct Grid {
    int3 L, N; /* subdomain size, grid size */
    float *d[NFIELDS];
};

struct Sampler {
    Grid dev, hst;
    long nsteps;
};

struct SampleDatum {
    long n;
    const Particle *pp;
    const float *ss;
};

enum {
    MAX_N_DATA = 10
};

struct SampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};

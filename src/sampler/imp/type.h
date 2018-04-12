enum {
    RHO, VX, VY, VZ,
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS_MAX
};

enum {
    NFIELDS_NO_STRESS   = VZ + 1,
    NFIELDS_WITH_STRESS = NFIELDS_MAX
};

static const char *names[NFIELDS_MAX] = {
    "density", "u", "v", "w",
    "sxx", "sxy", "sxz",
    "syy", "syz", "szz"
};

struct Grid {
    int3 L, N; /* subdomain size, grid size */
    float *d[NFIELDS_MAX];
    bool stress;
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

struct Datum_v {
    long n;
    const Particle *pp;
};

struct DatumS_v : Datum_v {
    const float *ss;
};

struct Grid {
    int3 L, N, M; /* subdomain size, grid size, margin */
    float4 *pp;   /* density, velocity                 */
    float *ss;    /* stresses                          */
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

enum {MAX_N_DATA};

struct SampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};

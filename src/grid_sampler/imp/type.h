enum {
    RHO, VX, VY, VZ,
    NFIELDS_P
};

enum {
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS_S
};

enum {
    NFIELDS_C = N_COLOR
};

enum {
    TOT_NFIELDS = NFIELDS_P + NFIELDS_S + NFIELDS_C
};

static const char *names_p[NFIELDS_P] =
    {"density", "u", "v", "w"};

static const char *names_s[NFIELDS_S] =
    {"sxx", "sxy", "sxz",
     "syy", "syz", "szz"};

#define make_str(a) #a ,
static const char *names_c[NFIELDS_C] =
    {XMACRO_COLOR(make_str)}; /* see inc/def.h */
#undef make_str

struct Grid {
    int3 L, N; /* subdomain size, grid size */
    float *p[NFIELDS_P]; /* particle density and velocity */
    float *s[NFIELDS_S]; /* stress                        */
    float *c[N_COLOR];   /* color density                 */
    bool stress;
};

struct GridSampler {
    Grid sdev, stdev, hst;
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

struct GridSampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};

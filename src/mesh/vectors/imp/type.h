enum {FLOAT, PARTICLE, ZERO};

struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
};

static void float_get(Vectors*, int i, float r[3]);
static void particle_get(Vectors*, int i, float r[3]);
static void zero_get(Vectors*, int i, float r[3]);
static void (*get[])(Vectors*, int i, float r[3]) = { float_get,  particle_get, zero_get };

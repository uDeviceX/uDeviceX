enum {FLOAT, PARTICLE};
struct Positions {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
};

static void float_get(Positions*, int i, float r[3]);
static void particle_get(Positions*, int i, float r[3]);
static void (*get[])(Positions*, int i, float r[3]) = { float_get,  particle_get };

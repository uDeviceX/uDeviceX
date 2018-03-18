enum {FLOAT, POSITIONS, ZERO};

struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
};

static void float_get(Vectors*, int i, float r[3]);
static void positions_get(Vectors*, int i, float r[3]);
static void zero_get(Vectors*, int i, float r[3]);
static void (*get[])(Vectors*, int i, float r[3]) = { float_get,  positions_get, zero_get };

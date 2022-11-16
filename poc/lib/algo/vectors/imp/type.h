struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
    Tform *tform;
};

static void float_get(const Vectors*, int i, float r[3]);

static void positions_get(const Vectors*, int i, float r[3]);
static void positions_edge_get(const Vectors*, int i, float r[3]);
static void positions_center_get(const Vectors*, int i, float r[3]);
static void velocities_get(const Vectors*, int i, float r[3]);
static void zero_get(const Vectors*, int i, float r[3]);

enum {FLOAT, POSITIONS, POSITIONS_EDGE, POSITIONS_CENTER, VELOCITIES, ZERO};
static void (*get[])(const Vectors*, int i, float r[3]) =
{ float_get,  positions_get, positions_edge_get, positions_center_get, velocities_get, zero_get };

typedef  void (*Local2Global)(const Coords *coords, float a[3], /**/ float b[3]);

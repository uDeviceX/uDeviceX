struct Scalars {
    int type;
    int n;
    union {
        const float  *ff;
        const double *dd;
    } D;
};

static double float_get (const Scalars*, int i);
static double double_get(const Scalars*, int i);
static double zero_get(const Scalars*, int i);

enum {FLOAT, DOUBLE, ZERO};
static double (*get[])(const Scalars*, int i) = { float_get,  double_get, zero_get};

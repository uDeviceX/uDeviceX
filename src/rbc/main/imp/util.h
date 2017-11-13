static void efaces(const char *f, int n0, /**/ int4 *faces) {
    /* get faces */
    int n;
    n = off::faces(f, n0, faces);
    if (n0 != n)
        ERR("wrong faces number in <%s> : %d != %d", f, n0, n);
}

static void evert(const char *f, int n0, /**/ float *vert) {
    /* get vertices */
    int n;
    n = off::vert(f, n0, vert);
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}

static void diff(float *a, float *b, /**/ float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}

static float vabs(float *a) {
    enum {X, Y, Z};
    float r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}

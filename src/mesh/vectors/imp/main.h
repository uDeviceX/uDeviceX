void vectors_float_ini(int n, const float *rr, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = FLOAT; q->n = n; q->D.rr = rr;
    *pq = q;
}

void vectors_postions_ini(int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = POSITIONS; q->n = n; q->D.pp = pp;
    *pq = q;
}

static void xl2xg3(const Coords *coords, float a[3], /**/ float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xg(coords, a[X]);
    b[Y] = yl2yg(coords, a[Y]);
    b[Z] = zl2zg(coords, a[Z]);
}
static void edge_ini_tform(const Coords *coords, Tform **pq) {
    enum {X, Y, Z};
    Tform *q;
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = a0[Y] = a0[Z] = 0;
    a1[X] = a1[Y] = a1[Z] = 1;
    xl2xg3(coords, a0, b0);
    xl2xg3(coords, a1, b1);
    tform_ini(&q);
    UC(tform_vector(a0, a1, b0, b1, /**/ q));
    *pq = q;
}
void vectors_postions_edge_ini(const Coords *coords, int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, &q->tform);
    *pq = q;
}

void vectors_velocities_ini(int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VELOCITIES; q->n = n; q->D.pp = pp;
    *pq = q;
}

void vectors_zero_ini(int n, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = ZERO; q->n = n;
    *pq = q;
}

void vectors_fin(Vectors *q) {
    tform_fin(q->tform);
    EFREE(q);
}

static void float_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void positions_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
static void positions_edge_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    UC(tform_convert(q->tform, pp[i].r, /**/ r));
}
static void positions_center_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    UC(tform_convert(q->tform, pp[i].r, /**/ r));
}
static void velocities_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].v[X];
    r[Y] = pp[i].v[Y];
    r[Z] = pp[i].v[Z];
}
static void zero_get(Vectors*, int, float r[3]) {
    enum {X, Y, Z};
    r[X] = r[Y] = r[Z] = 0;
}
void vectors_get(Vectors *q, int i, /**/ float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    get[q->type](q, i, r);
}

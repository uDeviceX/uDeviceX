void vectors_float_ini(int n, const float *rr, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = FLOAT; q->n = n; q->D.rr = rr;
    *pq = q;
}

void vectors_particle_ini(int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = PARTICLE; q->n = n; q->D.pp = pp;
    *pq = q;
}

void vectors_fin(Vectors *q) { EFREE(q); }

static void float_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void particle_get(Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
void vectors_get(Vectors *q, int i, /**/ float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    get[q->type](q, i, r);
}

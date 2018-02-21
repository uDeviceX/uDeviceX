void Positions_float_ini(int n, float *rr, /**/ Positions **pq) {
    Positions *q;
    EMALLOC(1, &q);
    q->type = PARTICLE; q->n = n; q->D.rr = rr;
    *pq = q;
}

void Positions_particle_ini(int n, Particle *pp, /**/ Positions **pq) {
    Positions *q;
    EMALLOC(1, &q);
    q->type = PARTICLE; q->n = n; q->D.pp = pp;
    *pq = q;
}

void Positions_fin(Positions *q) { EFREE(q); }

static void float_get(Positions *q, int i, float r[3]) {
    enum {X, Y, Z};
    float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void particle_get(Positions *q, int i, float r[3]) {
    enum {X, Y, Z};
    Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
void Positions_get(Positions *q, int i, /**/ float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    get[q->type](q, i, r);
}

void scalars_float_ini(int n, const float *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = scalars_impl::FLOAT; q->n = n; q->D.ff = rr;
    q->stamp = scalars_impl::MAGIC;
    *pq = q;
}

void scalars_double_ini(int n, const double *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = scalars_impl::DOUBLE; q->n = n; q->D.dd = rr;
    q->stamp = scalars_impl::MAGIC;
    *pq = q;
}

void scalars_vectors_ini(int n, const Vectors *vec, int dim, /**/ Scalars **pq) {
    enum {X, Y, Z};
    Scalars *q;
    EMALLOC(1, &q);
    switch (dim) {
    case X: q->type = scalars_impl::VECX; break;
    case Y: q->type = scalars_impl::VECY; break;
    case Z: q->type = scalars_impl::VECZ; break;
    default: ERR("wrong dim=%d", dim);
    }
    q->stamp = scalars_impl::MAGIC;
    q->n = n; q->D.vec = vec;
    *pq =q;
}

void scalars_zero_ini(int n, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = scalars_impl::ZERO; q->n = n;
    q->stamp = scalars_impl::MAGIC;
    *pq = q;
}


void scalars_one_ini(int n, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = scalars_impl::ONE; q->n = n;
    q->stamp = scalars_impl::MAGIC;
    *pq = q;
}

void scalars_fin(Scalars *q) { EFREE(q); }

namespace scalars_impl {
static double float_get(const Scalars *q, int i) { return q->D.ff[i]; }
static double double_get(const Scalars *q, int i) { return q->D.dd[i]; }
static double vecx_get(const Scalars *q, int i)   {
    enum {X};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
    return r[X];
}
static double vecy_get(const Scalars *q, int i)   {
    enum {X, Y};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
    return r[Y];
}
static double vecz_get(const Scalars *q, int i)   {
    enum {X, Y, Z};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
    return r[Z];
}

static double zero_get(const Scalars*, int) { return 0; }
static double one_get(const Scalars*, int) { return 1; }
}
double scalars_get(const Scalars *q, int i) {
    int n;
    if (q->stamp != scalars_impl::MAGIC)
        ERR("scalar is not initilized");
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    return scalars_impl::get[q->type](q, i);
}

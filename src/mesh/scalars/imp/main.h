void scalars_float_ini(int n, const float *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = FLOAT; q->n = n; q->D.ff = rr;
    *pq = q;
}

void scalars_double_ini(int n, const double *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = DOUBLE; q->n = n; q->D.dd = rr;
    *pq = q;
}

void scalars_zero_ini(int n, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = ZERO; q->n = n;
    *pq = q;
}

void scalars_fin(Scalars *q) { EFREE(q); }

static double float_get(const Scalars *q, int i) { return q->D.ff[i]; }
static double double_get(const Scalars *q, int i) { return q->D.dd[i]; }
static double zero_get(const Scalars*, int) { return 0; }
double scalars_get(const Scalars *q, int i) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    return get[q->type](q, i);
}

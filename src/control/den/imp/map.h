static int pred_circle(int i, int j, int k) {
    enum {X, Y, Z};
    float3 r, rc;
    float R;
    int *c, *d;
    c = m::coords;
    d = m::dims;

    rc.x = XS*(d[X]-2*c[X]-1)/2;
    rc.y = YS*(d[Y]-2*c[Y]-1)/2;
    rc.z = ZS*(d[Z]-2*c[Z]-1)/2;

    r.x = i - XS/2 + 0.5f - rc.x;
    r.y = j - YS/2 + 0.5f - rc.y;
    r.z = k - ZS/2 + 0.5f - rc.z;

    R = sqrt(r.x * r.x + r.y * r.y);
    return R < OUTFLOW_CIRCLE_R && R >= OUTFLOW_CIRCLE_R - 1;
}

static int predicate(int i, int j, int k) {
    int p = 0;
    if (OUTFLOW_CIRCLE)
        p = pred_circle(i, j, k);
    return p;
}

static void ini_map(int **ids, int *nids) {
    int *ii, i, j, k, n;
    size_t sz;
    
    sz = XS * YS * ZS * sizeof(int);
    UC(emalloc(sz, (void **) &ii));

    n = 0;
    for (k = 0; k < ZS; ++k) {
        for (j = 0; j < YS; ++j) {
            for (i = 0; i < XS; ++i) {
                if (predicate(i, j, k))
                    ii[n++] = i + XS * (j + YS * k);
            }
        }
    }
    
    *nids = n;
    sz = n * sizeof(int);
    
    CC(d::Malloc((void**) ids, sz));
    CC(d::Memcpy(*ids, ii, sz, H2D));
    
    UC(efree(ii));
}

void ini(DContMap **m0) {
    DContMap *m;
    
    UC(emalloc(sizeof(DContMap), (void**) m0));
    m = *m0;

    ini_map(&m->cids, &m->n);
}

void fin(DContMap *m) {
    CC(d::Free(m->cids));
    UC(efree(m));
}

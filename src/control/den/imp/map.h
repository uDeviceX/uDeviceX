static int pred_circle(Coords c, int i, int j, int k) {
    enum {X, Y, Z};
    float3 r;
    float R;

    r = make_float3(i, j, k);
    local2center(c, r, /**/ &r);
    
    R = sqrt(r.x * r.x + r.y * r.y);
    return R < OUTFLOW_CIRCLE_R && R >= OUTFLOW_CIRCLE_R - 1;
}

static int predicate(Coords c, int i, int j, int k) {
    int p = 0;
    if (OUTFLOW_CIRCLE)
        p = pred_circle(c, i, j, k);
    return p;
}

static void ini_map(Coords coords, int **ids, int *nids) {
    int *ii, i, j, k, n;
    size_t sz;
    
    sz = XS * YS * ZS * sizeof(int);
    UC(emalloc(sz, (void **) &ii));

    n = 0;
    for (k = 0; k < ZS; ++k) {
        for (j = 0; j < YS; ++j) {
            for (i = 0; i < XS; ++i) {
                if (predicate(coords, i, j, k))
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

void ini(Coords coords, DContMap **m0) {
    DContMap *m;
    
    UC(emalloc(sizeof(DContMap), (void**) m0));
    m = *m0;

    ini_map(coords, &m->cids, &m->n);
}

void fin(DContMap *m) {
    CC(d::Free(m->cids));
    UC(efree(m));
}

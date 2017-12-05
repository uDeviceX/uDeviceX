// TODO
static int predicate(int i, int j, int z) {return 1;}

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

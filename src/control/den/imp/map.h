struct Circle {
    float R;    
};

struct Plate {
    // TODO
};

struct None {};

static int predicate(const Coords *c, Circle p, int i, int j, int k) {
    enum {X, Y, Z};
    // TODO: for now assume centered at the center of the domain
    float3 r;
    float R;

    r = make_float3(i + (1 - XS) * 0.5f,
                    j + (1 - YS) * 0.5f,
                    k + (1 - ZS) * 0.5f);
    local2center(c, r, /**/ &r);
    
    R = sqrt(r.x * r.x + r.y * r.y);
    return R < p.R && R >= p.R - 1;
}

static int predicate(const Coords *c, Plate p, int i, int j, int k) {
    // TODO
    return 0;
}

static int predicate(const Coords *c, None p, int i, int j, int k) {
    return 0;
}


template <typename T>
static void ini_map(const Coords *coords, T p, int **ids, int *nids) {
    int *ii, i, j, k, n;
    size_t sz;
    
    sz = XS * YS * ZS * sizeof(int);
    UC(emalloc(sz, (void **) &ii));

    n = 0;
    for (k = 0; k < ZS; ++k) {
        for (j = 0; j < YS; ++j) {
            for (i = 0; i < XS; ++i) {
                if (predicate(coords, p, i, j, k))
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

static DContMap *alloc(DContMap **m0) {
    UC(emalloc(sizeof(DContMap), (void**) m0));
    return *m0;
}

void den_ini_map_none(const Coords *c, DContMap **m0) {
    DContMap *m = alloc(m0);
    None p;
    ini_map(c, p, &m->cids, &m->n);
}

void den_ini_map_circle(const Coords *c, float R, DContMap **m0) {
    DContMap *m = alloc(m0);
    Circle p;
    p.R = R;
    ini_map(c, p, &m->cids, &m->n);
}

void den_fin_map(DContMap *m) {
    CC(d::Free(m->cids));
    UC(efree(m));
}

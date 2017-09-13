// TODO: belongs to fragment?
static void estimates(int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

/* pinned allocation */

static void alloc_pinned_counts(int n, /**/ int **hc, int **dc) {
    CC(d::alloc_pinned((void**)hc, n * sizeof(int)));
    CC(d::HostGetDevicePointer((void**)dc, *hc, 0));
}

static void alloc_one_pinned_frag(int i, /**/ hBags *hb, dBags *db) {
    size_t n = hb->bsize * hb->capacity[i];
    CC(d::alloc_pinned(&hb->data[i], n));
    CC(d::HostGetDevicePointer(&db->data[i], hb->data[i], 0));
}

static void ini_pinned_bags(int nfrags, size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_pinned_frag(i, /**/ hb, db);
    alloc_pinned_counts(nfrags, /**/ &hb->counts, &db->counts);
}

void ini_pinned_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    ini_pinned_bags(NFRAGS, bsize, maxdensity, /**/ hb, db);
    hb->data[BULK] = db->data[BULK] = NULL;
}

void ini_pinned_full(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    ini_pinned_bags(NBAGS, bsize, maxdensity, /**/ hb, db);
}

/* normal allocation; counts are pinned */

static void alloc_one_frag(int i, /**/ hBags *hb, dBags *db) {
    size_t n = hb->bsize * hb->capacity[i];
    hb->data[i] = malloc(n);
    CC(d::Malloc(&db->data[i], n));
}

static void ini_bags(int nfrags, size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_frag(i, /**/ hb, db);
    alloc_pinned_counts(nfrags, /**/ &hb->counts, &db->counts);
}

void ini_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    ini_bags(NFRAGS, bsize, maxdensity, /**/ hb, db);
    hb->data[BULK] = db->data[BULK] = NULL;
}

void ini_full(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    ini_bags(NBAGS, bsize, maxdensity, /**/ hb, db);
}

/* normal allocation, host only */

static void alloc_one_frag(int i, /**/ hBags *hb) {
    size_t n = hb->bsize * hb->capacity[i];
    hb->data[i] = malloc(n);
}

static void ini_bags(int nfrags, size_t bsize, float maxdensity, /**/ hBags *hb) {
    hb->bsize = bsize;
    estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_frag(i, /**/ hb);
    hb->counts = (int*) malloc(nfrags * sizeof(int));
}

void ini_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb) {
    ini_bags(NFRAGS, bsize, maxdensity, /**/ hb);
    hb->data[BULK] = NULL;
}

void ini_full(size_t bsize, float maxdensity, /**/ hBags *hb) {
    ini_bags(NBAGS, bsize, maxdensity, /**/ hb);
}

/* stamp allocation */

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s) {
    int i, c, crd_rnk[3];
    
    for (i = 0; i < NFRAGS; ++i) {
        for (c = 0; c < 3; ++c)
            crd_rnk[c] = m::coords[c] +  frag_to_dir[i][c];
        MC(m::Cart_rank(comm, crd_rnk, s->ranks + i));
        s->tags[i] = frag_anti(i);
    }
    s->bt = get_tag(tg);
    MC(m::Comm_dup(comm, &s->cart));
}

/* pinned allocation */

static void alloc_counts(int n, /**/ int **hc) {
    *hc = (int*) malloc(n * sizeof(int));
}

/* generic allocation */
static void alloc_one_pair(int i, AllocMod mod, /**/ hBags *hb, dBags *db) {
    size_t n = hb->capacity[i] * hb->bsize;
    
    switch (mod) {
    case HST_ONLY:
        hb->data[i] = (data_t*) malloc(n);
        break;
    case DEV_ONLY:
        CC(d::Malloc((void**) &db->data[i], n));
        break;
    case PINNED:
        CC(d::alloc_pinned(&hb->data[i], n));
        CC(d::HostGetDevicePointer(&db->data[i], hb->data[i], 0));
        break;
    case NONE:
    default:
        break;
    }
}

void ini(AllocMod fmod, AllocMod bmod, size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    frag_estimates(NBAGS, maxdensity, hb->capacity);

    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        alloc_one_pair(i, fmod, /**/ hb, db);

    /* bulk */
    alloc_one_pair(frag_bulk, bmod, /**/ hb, db);

    alloc_counts(NBAGS, /**/ &hb->counts);
}



static void alloc_one_pinned_frag(int i, /**/ hBags *hb, dBags *db) {
    size_t n = hb->bsize * hb->capacity[i];
    CC(d::alloc_pinned(&hb->data[i], n));
    CC(d::HostGetDevicePointer(&db->data[i], hb->data[i], 0));
}

static void ini_pinned_bags(int nfrags, size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    frag_estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_pinned_frag(i, /**/ hb, db);
    alloc_counts(nfrags, /**/ &hb->counts);
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
    frag_estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_frag(i, /**/ hb, db);
    alloc_counts(nfrags, /**/ &hb->counts);
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
    frag_estimates(nfrags, maxdensity, hb->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_frag(i, /**/ hb);
    alloc_counts(nfrags, /**/ &hb->counts);
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
            crd_rnk[c] = m::coords[c] + frag_i2d(i,c);
        MC(m::Cart_rank(comm, crd_rnk, s->ranks + i));
        s->tags[i] = frag_anti(i);
    }
    s->bt = get_tag(tg);
    MC(m::Comm_dup(comm, &s->cart));
}

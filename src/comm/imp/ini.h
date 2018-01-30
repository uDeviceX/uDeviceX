/* pinned allocation */

static void alloc_counts(int n, /**/ int **hc) {
    size_t sz = n * sizeof(int);
    UC(emalloc(sz, (void**) hc));
    memset(*hc, 0, sz);
}

/* generic allocation */
static void alloc_pair(int i, AllocMod mod, /**/ hBags *hb, dBags *db) {
    size_t n = hb->capacity[i] * hb->bsize;
    
    switch (mod) {
    case HST_ONLY:
        if (n) UC(emalloc(n, (void**) &hb->data[i]));
        else   hb->data[i] = NULL;
        break;
    case DEV_ONLY:
        if (n) CC(d::Malloc((void**) &db->data[i], n));
        else db->data[i] = NULL;
        break;
    case PINNED:
        if (n) {
            CC(d::alloc_pinned(&hb->data[i], n));
            CC(d::HostGetDevicePointer(&db->data[i], hb->data[i], 0));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case PINNED_HST:
        if (n) CC(d::alloc_pinned(&hb->data[i], n));
        else hb->data[i] = NULL;
        break;
    case PINNED_DEV:
        if (n) {
            CC(d::alloc_pinned(&hb->data[i], n));
            CC(d::Malloc((void **) &db->data[i], n));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case NONE:
    default:
        break;
    }
}

int comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    memcpy(hb->capacity, capacity, NBAGS * sizeof(int));

    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        UC(alloc_pair(i, fmod, /**/ hb, db));

    /* bulk */
    UC(alloc_pair(frag_bulk, bmod, /**/ hb, db));

    UC(alloc_counts(NBAGS, /**/ &hb->counts));
    return 0;
}

int comm_ini(MPI_Comm cart, /**/ Comm **com_p) {
    int i, c, crd_rnk[3];
    int coords[3], periods[3], dims[3];
    Comm *com;

    UC(emalloc(sizeof(Comm), (void**) com_p));
    com = *com_p;
    
    MC(m::Cart_get(cart, 3, dims, periods, coords));
    
    for (i = 0; i < NFRAGS; ++i) {
        for (c = 0; c < 3; ++c)
            crd_rnk[c] = coords[c] + fraghst::i2d(i,c);
        MC(m::Cart_rank(cart, crd_rnk, com->ranks + i));
        com->tags[i] = fraghst::frag_anti(i);
    }
    MC(m::Comm_dup(cart, &com->cart));
    return 0;
}

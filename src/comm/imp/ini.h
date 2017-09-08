static void estimates(int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

static void alloc_one_pinned_frag(int i, /**/ Bags *b) {
    size_t n = b->bsize * b->capacity[i];
    CC(d::alloc_pinned(&b->hst[i], n));
    CC(d::HostGetDevicePointer(&b->dev[i], b->hst[i], 0));
}

static void ini_bags(int nfrags, size_t bsize, float maxdensity, /**/ Bags *b) {
    b->bsize = bsize;
    estimates(nfrags, maxdensity, b->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_pinned_frag(i, /**/ b);
}

void ini_no_bulk(size_t bsize, float maxdensity, /**/ Bags *b) {
    ini_bags(NFRAGS, bsize, maxdensity, /**/ b);
    b->hst[BULK] = b->dev[BULK] = NULL;
}

void ini_full(size_t bsize, float maxdensity, /**/ Bags *b) {
    ini_bags(NBAGS, bsize, maxdensity, /**/ b);
}


void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s) {
    int i, c, crd_rnk[3], crd_ank[3];
    
    for (i = 0; i < NFRAGS; ++i) {
        for (c = 0; c < 3; ++c) {
            crd_rnk[c] = m::coords[c] +  frag_to_dir[i][c];
            crd_ank[c] = m::coords[c] + frag_fro_dir[i][c];
        }
        MC(m::Cart_rank(comm, crd_rnk, s->rnks + i));
        MC(m::Cart_rank(comm, crd_ank, s->anks + i));
    }
    s->bt = get_tag(tg);
    MC(m::Comm_dup(comm, &s->cart));
}

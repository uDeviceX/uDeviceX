void emesh_unpack(int nv, const Unpack *u, /**/ int *nmhalo, Particle *pp) {
    int i, nm, n, s = 0, nmtot = 0;
    size_t sz;
    
    for (i = 0; i < NFRAGS; ++i) {
        nm = u->hpp.counts[i];
        n  = nm * nv; 
        sz = n * sizeof(Particle);
        if (nm) {
            CC(d::MemcpyAsync(pp + s, u->hpp.data[i], sz, H2D));
            KL(dev::shift_one_frag, (k_cnf(n)), (n, i, /**/ pp + s));
        }
        s += n;
        nmtot += nm;
    }
    *nmhalo = nmtot;
}

void emesh_get_num_frag_mesh(const Unpack *u, /**/ int cc[NFRAGS]) {
    memcpy(cc, u->hpp.counts, NFRAGS * sizeof(int));
}

/* momentum */
static void upload_bags(const hBags *h, dBags *d) {
    int i, c;
    data_t *src, *dst;
    size_t sz;
    for (i = 0; i < NFRAGS; ++i) {
        c = h->counts[i];
        sz = c * h->bsize;
        src = h->data[i];
        dst = d->data[i];
        if (sz)
            CC(d::MemcpyAsync(dst, src, sz, H2D));
    }
}

void emesh_upload(UnpackM *u) {
    upload_bags(&u->hmm, &u->dmm);
    upload_bags(&u->hii, &u->dii);
}

static int get_fragstarts(int nfrags, const int cc[], /**/ int *starts) {
    int i, s;
    starts[0] = s = 0;
    for (i = 0; i < nfrags; ++i)
        starts[i+1] = (s += cc[i]);
    return s;
}

void emesh_unpack_mom(int nt, const Pack *p, const UnpackM *u, /**/ Momentum *mm) {
    intp26 wrapii;
    Mop26 wrapmm;
    int27 fragstarts;
    int n;
    
    bag2Sarray(u->dmm, &wrapmm);
    bag2Sarray(u->dii, &wrapii);

    n = get_fragstarts(NFRAGS, u->hii.counts, /**/ fragstarts.d);
    
    KL(dev::unpack_mom, (k_cnf(n)), (nt, fragstarts, wrapii, wrapmm, p->map, /**/ mm));
}

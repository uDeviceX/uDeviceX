void unpack(int nv, const Unpack *u, /**/ int *nmhalo, Particle *pp) {
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
        CC(d::MemcpyAsync(dst, src, sz, H2D));
    }
}

void upload(UnpackM *u) {
    upload_bags(&u->hmm, &u->dmm);
    upload_bags(&u->hii, &u->dii);
}

static int accumulate(int n, const int *cc) {
    int i, s;
    for (i = s = 0; i < n; ++i) s += cc[i];
    return s;
}

static int get_fragstarts(int nfrags, const int *cc[], const int ncc[], /**/ int *starts) {
    int i, c, s;
    starts[0] = s = 0;
    for (i = 0; i < nfrags; ++i) {
        c = accumulate(ncc[i], cc[i]);
        starts[i+1] = (s += c);
    }
    return s;
}

void unpack_mom(int nt, const Pack *p, const UnpackM *u, /**/ Momentum *mm) {
    intp26 wrapii;
    Mop26 wrapmm;
    int27 fragstarts;
    int n;
    
    bag2Sarray(u->dmm, &wrapmm);
    bag2Sarray(u->dii, &wrapii);

    n = get_fragstarts(NFRAGS, (const int**) u->hcc.data, u->hcc.counts, /**/ fragstarts.d);
    
    KL(dev::unpack_mom, (k_cnf(n)), (nt, fragstarts, wrapii, wrapmm, p->map, /**/ mm));
}

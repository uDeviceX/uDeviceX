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

void unpack_mom(const Pack *p, const UnpackM *u) {

}

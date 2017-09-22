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

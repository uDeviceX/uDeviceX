
// TODO smarter bulk
static int unpack_pp(int nv, const hBags bags, /**/ Particle *pp) {
    int s, c;
    size_t bs = bags.bsize;

    /* bulk */
    c = bags.counts[frag_bulk];
    if (c) CC(d::MemcpyAsync(pp + s, bags.data[frag_bulk], c * bs, D2D));

    s = c * nv;
    
    /* collect fragments */ 
    for (i = 0; i < NFRAGS; ++i) {
        c = bags.counts[i];
        if (c) CC(d::MemcpyAsync(pp + s, bags.data[i], c * bs, H2D));
        s += c * nv;
    }

    /* shift */
    // TODO
}

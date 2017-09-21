void upload(Unpack *u) {
    int i, c;
    size_t sz;
    data_t *src, *dst;

    for (i = 0; i < NFRAGS; ++i) {
        c = u->hpp.counts[i];
        if (c) {
            sz = c * u->hpp.bsize;
            dst = u->dpp.data[i];
            src = u->hpp.data[i];
            CC(d::MemcpyAsync(dst, src, sz, H2D));
        }
    }
}

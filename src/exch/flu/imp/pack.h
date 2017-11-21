void download_cell_starts(intp26 src, /**/ intp26 dst) {
    int i, nc;
    size_t sz;

    for(i = 0; i < NFRAGS; ++i) {
        nc = frag_ncell(i);
        sz = (nc + 1) * sizeof(int);
        d::MemcpyAsync(dst.d[i], src.d[i], sz, D2H);
    }
}


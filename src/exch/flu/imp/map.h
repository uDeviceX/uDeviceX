static int get_cell_num(int26 cc) {
    for (int i = 0; i < NFRAGS; ++i) cc.d[i] = frag_ncell(i);
    return cc.d[NFRAGS];
}

static void scan(int26 cc, /**/ int27 ss) {
    ss.d[0] = 0;
    for (int i = 0; i < 26; ++i) ss.d[i+1] = ss.d[i] + cc.d[i];
}

void compute_map(const int *start, const int *count, /**/ Pack *p) {
    int nc;
    int26 cc;
    int27 ss;
    nc = get_cell_num(/**/ cc);
    scan(cc, /**/ ss);
    KL(dev::count, (k_cnf(nc)), (ss, start, count, /**/ p->bss, p->bcc));
    KL(dev::scan<32>, (26, 32 * 32), (cc, p->bcc, /**/ p->fss));
}


void download_cell_starts(intp26 src, /**/ intp26 dst) {
    int i, nc;
    size_t sz;

    for(i = 0; i < NFRAGS; ++i) {
        nc = frag_ncell(i);
        sz = (nc + 1) * sizeof(int);
        d::MemcpyAsync(dst.d[i], src.d[i], sz, D2H);
    }
}


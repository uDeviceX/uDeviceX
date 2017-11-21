static int get_cell_num(int *cc) {
    int i, nc, c;
    for (nc = i = 0; i < NFRAGS; ++i) {
        c = cc[i] = frag_ncell(i);
        nc += c;
    }
    return nc;
}

static void scan(int n, int *cc, /**/ int *ss) {
    ss[0] = 0;
    for (int i = 0; i < n; ++i) ss[i+1] = ss[i] + cc[i];
}

void compute_map(const int *start, const int *count, /**/ Pack *p) {
    int nc;
    int26 cc;
    int27 ss;
    intp26 fss;
    nc = get_cell_num(/**/ cc.d);
    scan(NFRAGS, cc.d, /**/ ss.d);
    KL(dev::count_cells, (k_cnf(nc)), (ss, start, count, /**/ p->bss, p->bcc));

    bag2Sarray(p->dfss, /**/ &fss);
    KL(dev::scan<32>, (26, 32 * 32), (cc, p->bcc, /**/ fss));
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


static int get_cell_num(int3 L, int *cc) {
    int i, nc, c;
    for (nc = i = 0; i < NFRAGS; ++i) {
        c = cc[i] = fraghst::ncell(L, i) + 1;
        nc += c;
    }
    return nc;
}

static void scan(int n, int *cc, /**/ int *ss) {
    ss[0] = 0;
    for (int i = 0; i < n; ++i) ss[i+1] = ss[i] + cc[i];
}

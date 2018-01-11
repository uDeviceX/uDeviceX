void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p) {
    int nc;
    int26 cc;
    int27 ss;
    nc = get_cell_num(/**/ cc.d);
    scan(NFRAGS, cc.d, /**/ ss.d);
    KL(dev::count_cells, (k_cnf(nc)), (ss, start, count, /**/ p->bss, p->bcc));    
    KL(dev::scan<32>, (26, 32 * 32), (cc, p->bcc, /**/ p->fss));
}


static void download_cell_starts(intp26 src, /**/ intp26 dst) {
    int i, nc;
    size_t sz;

    for(i = 0; i < NFRAGS; ++i) {
        nc = frag_ncell(i) + 1;
        sz = nc * sizeof(int);
        CC(d::MemcpyAsync(dst.d[i], src.d[i], sz, D2H));
    }
}

void eflu_download_cell_starts(/**/ EFluPack *p) {
    intp26 fss_hst;
    bag2Sarray(p->hfss, /**/ &fss_hst);
    download_cell_starts(p->fss, /**/ fss_hst);
    /* size of the messages is fixed throughout the whole simulation */
    /* sizes are frag_ncell(fid) + 1 */
}


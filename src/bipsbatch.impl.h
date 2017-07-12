namespace bipsbatch {
static void get_start(Frag frag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)frag[i].ndst + 15) / 16);
}

void interactions(Frag frag_hst[26], /**/ float *ff) {
    static unsigned int start_hst[27];
    int nt; /* threads */
    get_start(frag_hst, /**/ start_hst);
    nt = 2 * start_hst[26];
    
    CC(cudaMemcpyToSymbolAsync(ffrag_dev, frag_hst, sizeof(Frag) * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(start_dev, start_hst, sizeof(start_hst), 0, H2D));
    dSync();
    if (nt) k_bipsbatch::force<<<k_cnf(nt)>>>(ffrag_dev, start_dev, ff);
}
}

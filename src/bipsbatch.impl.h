namespace bipsbatch {
static void get_start(Frag frag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)frag[i].ndst + 15) / 16);
}

void interactions(Frag ffrag_hst[26], Rnd rrnd_hst[26], /**/ float *ff) {
    static unsigned int start_hst[27];
    int n; /* number of threads */
    get_start(ffrag_hst, /**/ start_hst);
    n = 2 * start_hst[26];
    
    CC(cudaMemcpyToSymbolAsync(k_bipsbatch::ffrag, ffrag_hst, sizeof(Frag) * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_bipsbatch::start, start_hst, sizeof(start_hst), 0, H2D));
    dSync();
    if (n) k_bipsbatch::force<<<k_cnf(n)>>>(ff);
}
}

namespace bipsbatch {
static void get_start(Frag frag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)frag[i].ndst + 15) / 16);
}

void interactions(SFrag ssfrag[], Frag ffrag[], Rnd rrnd[], /**/ float *ff) {
    static unsigned int start[27];
    int n; /* number of threads */
    get_start(ffrag, /**/ start);
    n = 2 * start[26];
    
    CC(cudaMemcpyToSymbolAsync(k_bipsbatch::start, start, sizeof(start), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_bipsbatch::ffrag, ffrag, sizeof(Frag) * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_bipsbatch::rrnd,   rrnd, sizeof(Rnd) * 26,  0, H2D));

    dSync();
    if (n) k_bipsbatch::force<<<k_cnf(n)>>>(ff);
}
}

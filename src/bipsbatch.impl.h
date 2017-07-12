namespace bipsbatch {
void interactions(BatchInfo infos[26], /**/ float *ff) {
    CC(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 26, 0, H2D));
    static unsigned int hstart_padded[27];
    int nt; /* threads */
    int i;
    hstart_padded[0] = 0;
    for (i = 0; i < 26; ++i) hstart_padded[i + 1] = hstart_padded[i] + 16 * (((unsigned int)infos[i].ndst + 15) / 16);
    CC(cudaMemcpyToSymbolAsync(start, hstart_padded, sizeof(hstart_padded), 0, H2D));
    nt = 2 * hstart_padded[26];
    dSync();
    if (nt) force<<<k_cnf(nt)>>>(ff);
}
}

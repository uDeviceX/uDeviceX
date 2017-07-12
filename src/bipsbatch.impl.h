namespace bipsbatch {
void interactions(BatchInfo infos[26], /**/ float *acc) {

    CC(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 26, 0,
                               H2D));

    static unsigned int hstart_padded[27];

    hstart_padded[0] = 0;
    for (int i = 0; i < 26; ++i)
    hstart_padded[i + 1] =
        hstart_padded[i] + 16 * (((unsigned int)infos[i].ndst + 15) / 16);

    CC(cudaMemcpyToSymbolAsync(start, hstart_padded, sizeof(hstart_padded), 0,
                               H2D));

    int nthreads = 2 * hstart_padded[26];

    dSync();

    if (nthreads)
    interaction_kernel<<<k_cnf(nthreads)>>> (acc);
}

}

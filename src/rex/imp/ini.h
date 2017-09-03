namespace sub {
static void ini_copy(LFrag *local) {
    int i;
    for (i = 0; i < 26; ++i)
        CC(cudaMemcpyToSymbol(dev::g::indexes, &local[i].indexes, sizeof(int*), sizeof(int*) * i, H2D));
}

static void ini_ff(LFrag *local) {
    int i;
    float *ff[26];
    for (i = 0; i < 26; ++i) ff[i] = (float*)local[i].ff;
    CC(cudaMemcpyToSymbolAsync(dev::g::ff, ff, sizeof(ff), 0, H2D));
}

void ini(LFrag *local) {
    ini_copy(local);
    ini_ff(local);
}
}

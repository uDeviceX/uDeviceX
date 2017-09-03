namespace rex {
static int i2max(int i) { /* fragment id to maximum size */
    return MAX_OBJ_DENSITY*frag_ncell(i);
}

static void ini_local() {
    int i, n;
    LFrag *h;
    for (i = 0; i < 26; i++) {
        n = i2max(i);
        h = &local[i];
        Dalloc(&h->indexes, n);

        Palloc0(&h->ff_pi, n);
        Link(&h->ff, h->ff_pi);
    }
}

static void ini_copy() {
    int i;
    for (i = 0; i < 26; ++i)
        CC(cudaMemcpyToSymbol(k_rex::g::indexes, &local[i].indexes, sizeof(int*), sizeof(int*) * i, H2D));
}

static void ini_ff() {
    int i;
    float *ff[26];
    for (i = 0; i < 26; ++i) ff[i] = (float*)local[i].ff;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::ff, ff, sizeof(ff), 0, H2D));
}

void ini() {
    ini_local();
    ini_copy();
    ini_ff();
}
}

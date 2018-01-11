static void ini0(RbcRnd *d, int n, long seed) {
    Dalloc(&d->r, n);
    CU(api::CreateGenerator(&d->g));
    seed = decode_seed(seed);
    CU(api::SetPseudoRandomGeneratorSeed(d->g,  seed));
    d->max = n;
}
void ini(RbcRnd **pd, int n, long seed) {
    RbcRnd* d;
    UC(emalloc(sizeof(RbcRnd), (void**) &d));
    ini0(d, n, seed);
    *pd = d;
}

static void fin0(RbcRnd *d) {
    Dfree(d->r);
    CU(api::DestroyGenerator(d->g));
}
void fin(RbcRnd *d) {
    fin0(d);
    free(d);
}

static void assert_n(int n, int max, const char *s) {
    if (n > max)
        ERR("%s: n = %d > max = %d", s, n , max);
}

void gen(RbcRnd *d, int n) {
    assert_n(n, d->max, "rbc::rnd::gen");
    api::GenerateNormal(d->g, d->r, n);
}

float get_hst(const RbcRnd *d, int i) {
    float x;
    float *r;
    assert_n(i, d->max, "rbc::rnd::get_hst");
    r = d->r;
    cD2H(&x, &r[i], 1);
    return x;
}

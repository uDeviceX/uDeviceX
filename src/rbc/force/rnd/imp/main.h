static void ini0(RbcRnd *d, int n, long seed) {
    Dalloc(&d->r, n);
    CU(rnd_api::CreateGenerator(&d->g));
    seed = decode_seed(seed);
    CU(rnd_api::SetPseudoRandomGeneratorSeed(d->g,  seed));
    d->max = n;
}
void rbc_rnd_ini(int n, long seed, RbcRnd **pq) {
    RbcRnd *q;
    EMALLOC(1, &q);
    UC(ini0(q, n, seed));
    *pq = q;
}

static void fin0(RbcRnd *d) {
    Dfree(d->r);
    CU(rnd_api::DestroyGenerator(d->g));
}
void rbc_rnd_fin(RbcRnd *d) {
    fin0(d);
    EFREE(d);
}

static void assert_n(int n, int max, const char *s) {
    if (n > max)
        ERR("%s: n = %d > max = %d", s, n , max);
}

void rbc_rnd_gen(RbcRnd *d, int n, float **pq) {
    assert_n(n, d->max, "rbc::rnd::gen");
    rnd_api::GenerateNormal(d->g, d->r, n);
    *pq = d->r;
}

float rbc_rnd_get_hst(const RbcRnd *d, int i) {
    float x;
    float *r;
    assert_n(i, d->max, "rbc::rnd::get_hst");
    r = d->r;
    cD2H(&x, &r[i], 1);
    return x;
}

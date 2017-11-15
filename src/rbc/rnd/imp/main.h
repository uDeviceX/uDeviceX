static long decode_seed_env() {
    char *s;
    s = getenv("RBC_RND");
    if   (s == NULL) return 0;
    else             return atol(s);
}
static long decode_seed_time() {
    long t;
    t = os::time();
    MSG("t: %ld", t);
    return t;
}
static int decode_seed(long seed) {
    if      (seed == ENV )  return decode_seed_env();
    else if (seed == TIME) return decode_seed_time();
    else return seed;
}
static void ini0(D *d, int n, long seed) {
    Dalloc(&d->r, n);
    CU(curandCreateGenerator(&d->g, CURAND_RNG_PSEUDO_DEFAULT));
    seed = decode_seed(seed);
    CU(curandSetPseudoRandomGeneratorSeed(d->g,  seed));
    d->max = n;
}
void ini(D **pd, int n, int seed) {
    D* d;
    d = (D*)malloc(sizeof(D));
    ini0(d, n, seed);
    *pd = d;
}

static void fin0(D *d) {
    Dfree(d->r);
    CU(curandDestroyGenerator(d->g));
}
void fin(D *d) {
    fin0(d);
    free(d);
}

static void assert_n(int n, int max, const char *s) {
    if (n > max)
        ERR("%s: n = %d > max = %d", s, n , max);
}

void gen(D *d, int n) {
    assert_n(n, d->max, "rbc::rnd::gen");
    float mean, std;
    mean = 0; std = 1;
    CU(curandGenerateNormal(d->g, d->r, n, mean, std));
}

float get_hst(const D *d, int i) {
    float x;
    float *r;
    assert_n(i, d->max, "rbc::rnd::get_hst");
    r = d->r;
    cD2H(&x, &r[i], 1);
    return x;
}

static long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

static int get_nfields(const Grid *g) {
    return g->stress ?
        NFIELDS_WITH_STRESS :
        NFIELDS_NO_STRESS;
}

static void ini_dev_grid(bool stress, int3 L, int3 N, Grid *g) {
    long i, n;
    g->N = N;
    g->L = L;
    g->stress = stress;

    n = get_size(g);
    for (i = 0; i < get_nfields(g); ++i)
        Dalloc(&g->d[i], n);
}

static void ini_hst_grid(bool stress, int3 L, int3 N, Grid *g) {
    long i, n;
    g->N = N;
    g->L = L;
    g->stress = stress;

    n = get_size(g);
    for (i = 0; i < get_nfields(g); ++i)
        EMALLOC(n,  &g->d[i]);
}

static void fin_dev_grid(Grid *g) {
    for (int i = 0; i < get_nfields(g); ++i)
        Dfree(g->d[i]);
}

static void fin_hst_grid(Grid *g) {
    for (int i = 0; i < get_nfields(g); ++i)
        EFREE(g->d[i]);
}

void sampler_ini(bool stress, int3 L, int3 N, Sampler **s0) {
    Sampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(stress, L, N, &s->dev));
    UC(ini_hst_grid(stress, L, N, &s->hst));
    UC(sampler_reset(s));
}

void sampler_fin(Sampler *s) {
    UC(fin_dev_grid(&s->dev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_grid(Grid *g) {
    long i, n = get_size(g);
    for (i = 0; i < get_nfields(g); ++i)
        DzeroA(g->d[i], n);
}

void sampler_reset(Sampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->dev));
}

void sampler_add(const SampleData *data, Sampler *s) {
    long i, n;
    SampleDatum d;
    Grid g = s->dev;
    for (i = 0; i < data->n; ++i) {
        d = data->d[i];
        n = d.n;
        KL(sampler_dev::add, (k_cnf(n)), (d, g));
    }
    ++ s->nsteps;
}

static void avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::avg, (k_cnf(n)), (nsteps, *g));
}

static void download(const Grid *dev, Grid *hst) {
    long i, n = get_size(dev);
    for (i = 0; i < get_nfields(dev); ++i)
        aD2H(dev->d[i], hst->d[i], n);
}

void sampler_dump(Sampler *s) {
    UC(avg(s->nsteps, &s->dev));
    UC(download(&s->dev, &s->hst));
    
}

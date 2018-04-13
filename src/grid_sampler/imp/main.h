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

void grid_sampler_ini(bool stress, int3 L, int3 N, GridSampler **s0) {
    GridSampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(stress, L, N, &s->dev));
    UC(ini_hst_grid(stress, L, N, &s->hst));
    UC(grid_sampler_reset(s));
}

void grid_sampler_fin(GridSampler *s) {
    UC(fin_dev_grid(&s->dev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_grid(Grid *g) {
    long i, n = get_size(g);
    for (i = 0; i < get_nfields(g); ++i)
        DzeroA(g->d[i], n);
}

void grid_sampler_reset(GridSampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->dev));
}

static void datum_view(const SampleDatum *d, Datum_v *v) {
    v->n = d->n;
    v->pp = d->pp;
}

static void datum_view(const SampleDatum *d, DatumS_v *v) {
    v->n = d->n;
    v->pp = d->pp;
    v->ss = d->ss;
}

template <typename Datum>
static void add(const GridSampleData *data, Grid g) {
    long i, n;
    const SampleDatum *d;
    Datum v;
    for (i = 0; i < data->n; ++i) {
        d = &data->d[i];
        datum_view(d, &v);
        n = v.n;
        KL(sampler_dev::add, (k_cnf(n)), (v, g));
    }
}

void grid_sampler_add(const GridSampleData *data, GridSampler *s) {
    Grid g = s->dev;
    if (g.stress)
        add<DatumS_v>(data, g);
    else
        add<Datum_v>(data, g);
    s->nsteps ++;
}

static void avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::avg, (k_cnf(n)), (nsteps, *g));
}

static void download(const Grid *dev, Grid *hst) {
    long i, n = get_size(dev);
    for (i = 0; i < get_nfields(dev); ++i)
        aD2H(hst->d[i], dev->d[i], n);
}

static void dump(MPI_Comm cart, const char *dir, long id, const Grid *g) {
    int ncmp = get_nfields(g);
    const float **data = (const float**) g->d;
    UC(grid_write(g->N, g->L, cart, dir, id, ncmp, data, names));
}

void grid_sampler_dump(MPI_Comm cart, const char *dir, long id, GridSampler *s) {
    UC(avg(s->nsteps, &s->dev));
    UC(download(&s->dev, &s->hst));
    dSync();
    UC(dump(cart, dir, id, &s->hst));
}

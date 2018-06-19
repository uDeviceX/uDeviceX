static long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

static void dev_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) Dalloc(&a[i], n);
}

static void ini_dev_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);

    UC(             dev_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( dev_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( dev_alloc_n(NFIELDS_S, n, g->s) );
}

static void hst_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) EMALLOC(n, &a[i]);
}

static void ini_hst_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);
    UC(             hst_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( hst_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( hst_alloc_n(NFIELDS_S, n, g->s) );
}

static void dev_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) Dfree(a[i]);
}

static void fin_dev_grid(Grid *g) {
    UC(                dev_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( dev_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( dev_free_n(NFIELDS_S, g->s) );
}

static void hst_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) EFREE(a[i]);
}

static void fin_hst_grid(Grid *g) {
    UC(                hst_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( hst_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( hst_free_n(NFIELDS_S, g->s) );
}

void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler **s0) {
    GridSampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(colors, stress, L, N, &s->sdev));
    UC(ini_dev_grid(colors, stress, L, N, &s->stdev));
    UC(ini_hst_grid(colors, stress, L, N, &s->hst));
    UC(grid_sampler_reset(s));
}

void grid_sampler_fin(GridSampler *s) {
    UC(fin_dev_grid(&s->sdev));
    UC(fin_dev_grid(&s->stdev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) DzeroA(a[i], n);
}

static void reset_dev_grid(Grid *g) {
    long n = get_size(g);
    UC(                reset_dev_n(NFIELDS_P, n, g->p) );
    if (g->colors) UC( reset_dev_n(NFIELDS_C, n, g->c) );
    if (g->stress) UC( reset_dev_n(NFIELDS_S, n, g->s) );
}

void grid_sampler_reset(GridSampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->stdev));
}

static void add_data_to_grid(const GridSampleData *data, Grid *g) {
    long i, n;
    SampleDatum d;
    for (i = 0; i < data->n; ++i) {
        d = data->d[i];
        n = d.n;
        KL(sampler_dev::add, (k_cnf(n)), (d, *g));
    }
}

static void space_avg(Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::space_avg, (k_cnf(n)), (*g));
}

static void add_to_grid(const Grid *s, Grid *st) {
    long n = get_size(s);
    KL(sampler_dev::add_to_grid, (k_cnf(n)), (*s, *st));
}

void grid_sampler_add(const GridSampleData *data, GridSampler *s) {
    Grid *sg, *stg;
    sg  = &s->sdev;
    stg = &s->stdev;
    UC(reset_dev_grid(sg));
    UC(add_data_to_grid(data, sg));
    UC(space_avg(sg));
    UC(add_to_grid(sg, stg));
    s->nsteps ++;
}

static void time_avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::time_avg, (k_cnf(n)), (nsteps, *g));
}

static void download_n(int na, long n, float * const dev[], float *hst[]) {
   for (int i = 0; i < na; ++i) aD2H(hst[i], dev[i], n);
}

static void download(const Grid *dev, Grid *hst) {
    long n = get_size(dev);
    UC(                  download_n(NFIELDS_P, n, dev->p, hst->p) );
    if (dev->colors) UC( download_n(NFIELDS_C, n, dev->c, hst->c) );
    if (dev->stress) UC( download_n(NFIELDS_S, n, dev->s, hst->s) );
}

static void build_desc(int ncmp, const char * const name[], const float * const data[],
                       int *j, const char *dst_name[], const float *dst_data[]) {
    for (int i = 0; i < ncmp; ++i) {
        dst_data[*j] = data[i];
        dst_name[*j] = name[i];
        ++(*j);
    }
}

static void dump(MPI_Comm cart, const char *dir, long id, const Grid *g) {
    char path[FILENAME_MAX] = {'\0'};
    int ncmp;
    const float *data[TOT_NFIELDS];
    const char *names[TOT_NFIELDS];

    ncmp = 0;
    build_desc               (NFIELDS_P, names_p, g->p, /**/ &ncmp, names, data);
    if (g->colors) build_desc(NFIELDS_C, names_c, g->c, /**/ &ncmp, names, data);
    if (g->stress) build_desc(NFIELDS_S, names_s, g->s, /**/ &ncmp, names, data);
    
    sprintf(path, DUMP_BASE "/%s/%04ld.h5", dir, id);
    
    UC(grid_write(g->N, g->L, cart, path, ncmp, data, names));
}

void grid_sampler_dump(MPI_Comm cart, const char *dir, long id, GridSampler *s) {
    UC(time_avg(s->nsteps, &s->stdev));
    UC(download(&s->stdev, &s->hst));
    dSync();
    UC(dump(cart, dir, id, &s->hst));
}

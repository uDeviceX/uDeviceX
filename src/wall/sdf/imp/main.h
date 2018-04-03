static int3 texture_grid_size(int3 L, int3 M) {
    enum {PAD = 16};
    int3 T;
    int x, y, z; /* grid extents in real space */
    int ty, tz;
    
    x = L.x + 2 * M.x;
    y = L.y + 2 * M.y;
    z = L.z + 2 * M.z;
    
    T.x = PAD * PAD;
    ty = ceiln(y * T.x, x);
    tz = ceiln(z * T.x, x);
    T.y = PAD * ceiln(ty, PAD);
    T.z = PAD * ceiln(tz, PAD);

    return T;
};

void sdf_ini(int3 L, Sdf **pq) {
    Sdf *q;
    int3 Lte, M;
    M = make_int3(XWM, YWM, ZWM);
    Lte = texture_grid_size(L, M);
    EMALLOC(1, &q);
    UC(array3d_ini(&q->arr, Lte.x, Lte.y, Lte.z));
    UC(  tform_ini(&q->t));

    q->Lte = Lte;
    q->cheap_threshold = - sqrt(3.f) * ((float) (L.x + 2*M.x) / (float)Lte.x);

    *pq = q;
}

void sdf_fin(Sdf *q) {
    UC(array3d_fin(q->arr));
    UC(  tex3d_fin(q->tex));
    UC(  tform_fin(q->t));
    EFREE(q);
}

void sdf_bounce(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    UC(bounce_back(dt, wv, c, sdf, n, /**/ pp));
}


void sdf_to_view(const Sdf *q, /**/ Sdf_v *v) {
    tex3d_to_view(q->tex, &v->tex);
    tform_to_view(q->t  , &v->t);
    v->cheap_threshold = q->cheap_threshold;
}

enum {NTHREADS = 128};

static void gen_rnd(long n, float *UU) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, UU, n);
    curandDestroyGenerator(gen);
}

static void chunk_counts(int3 L, const Sdf *sdf, int nsamples, int nchunks, int *counts_hst) {
    int *counts_dev;
    Sdf_v view;
    float *UU;
    sdf_to_view(sdf, &view);

    Dalloc(&UU, 3 * nsamples);
    Dalloc(&counts_dev, nchunks);    
    DzeroA(counts_dev, nchunks);

    gen_rnd(3 * nsamples, UU);
    
    KL(sdf_dev::count_inside,
       (nchunks, NTHREADS),
       (view, L, nsamples, UU, /**/ counts_dev));

    cH2D(counts_dev, counts_hst, nsamples);
    Dfree(counts_dev);
    Dfree(UU);
}

static long reduce(int n, int *d) {
    long c, i;
    for (i = c = 0; i < n; ++i) c += d[i];
    return c;
}

static long subdomain_counts(int3 L, const Sdf *sdf, long nsamples) {
    int *counts_hst;
    long nin;
    int nchunks;

    nchunks = ceiln(nsamples, NTHREADS);
    EMALLOC(nchunks, &counts_hst);

    chunk_counts(L, sdf, nsamples, nchunks, counts_hst);
    nin = reduce(nchunks, counts_hst);
    
    EFREE(counts_hst);
    return nin;
}

static double subdomain_volume(int3 L, long nin, long nsamples) {
    double V = (double) nin / (double) nsamples;
    V *= L.x * L.y * L.z;
    return V;
}

double sdf_compute_volume(MPI_Comm comm, int3 L, const Sdf *sdf, long nsamples) {
    double loc, tot = 0;
    long nin;

    nin = subdomain_counts(L, sdf, nsamples);
    loc = subdomain_volume(L, nin, nsamples);

    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, comm));

    return tot;
}

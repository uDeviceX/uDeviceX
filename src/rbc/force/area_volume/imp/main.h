static void assert_tri(int nt, int nv, const int4 *tri) {
    int *rank;
    int f0, f1, f2, i;
    int4 f;
    UC(emalloc(nv*sizeof(rank), (void**)&rank));
    for (i = 0; i < nv; i++) rank[i] = 0;
    for (i = 0; i < nt; i++) {
        f = tri[i];
        f0 = f.x; f1 = f.y; f2 = f.z;
        if (f0 < 0 || f0 >= nv || f1 < 0 || f1 >= nv || f2 < 0 || f2 >= nv)
            ERR("wrong triangle: %d = [%d %d %d] (v = %d, t = %d)", i, f0, f1, f2, nv, nt);
        rank[f0]++; rank[f1]++; rank[f2]++;
    }
    for (i = 0; i < nv; i++)
        if (rank[i] == 0) ERR("isolated vertex: %d (v = %d, t = %d)", i, nv, nt);
    EFREE(rank);
}

void area_volume_ini(int nv, int nt, const int4 *hst, int max_cell, /**/ AreaVolume **pq) {
    AreaVolume *q;
    EMALLOC(1, &q);
    q->nt = nt; q->nv = nv; q->max_cell = max_cell; q->Computed = 0;
    Dalloc(&q->tri, nt);
    Dalloc(&q->av , 2*max_cell);
    EMALLOC(2*max_cell, &q->av_hst);
    UC(assert_tri(nt, nv, hst));
    cH2D(q->tri, hst, nt);
    *pq = q;
}

void area_volume_fin(AreaVolume *q) {
    Dfree(q->tri);
    Dfree(q->av);
    EFREE(q->av_hst);
    EFREE(q);
}

static void compute(int nv, int nt, int nc, const Particle *pp, const int4 *tri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(rbc_force_av_dev::main, (avBlocks, avThreads), (nt, nv, pp, tri, av));
}

void area_volume_compute(AreaVolume *q, int nc, const Particle *pp, /**/ float **pav) {
    if (nc > q->max_cell) ERR("nc=%d > max_cell=%d", nc, q->max_cell);
    if (!d::is_device_pointer(pp)) ERR("`pp`  is not a device pointer");
    UC(compute(q->nv, q->nt, nc, pp, q->tri, q->av));
    q->Computed = 1; q->nc = nc;
    *pav = q->av;
}

const int4* area_volume_tri(AreaVolume *q) { return q->tri; }

void area_volume_host(AreaVolume *q, /**/ float **pav) {
    if (!q->Computed) ERR("call `area_volume_host` before `area_volume_compute`");
    cD2H(q->av_hst, q->av, 2*q->nc);
    *pav = q->av_hst;
}

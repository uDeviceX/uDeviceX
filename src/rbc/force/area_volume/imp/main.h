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
    UC(efree(rank));
}

void area_volume_ini(int nv, int nt, const int4 *hst, /**/ AreaVolume **pq) {
    AreaVolume *q;
    UC(emalloc(sizeof(AreaVolume), (void**)&q));
    q->nt = nt; q->nv = nv;
    Dalloc(&q->tri, nt);
    UC(assert_tri(nt, nv, hst));
    cH2D(q->tri, hst, nt);
    *pq = q;
}

void area_volume_fin(AreaVolume *q) {
    Dfree(q->tri);
    UC(efree(q));
}

static void compute(int nt, int nv, int nc, const Particle *pp, const int4 *tri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(dev::main, (avBlocks, avThreads), (nt, nv, pp, tri, av));
}

void area_volume_compute(AreaVolume *q, int nc, const Particle *pp, /**/ float *av) {
    UC(compute(q->nt, q->nv, nc, pp, q->tri, av));
}

int4* area_volume_tri(AreaVolume *q) { return q->tri; }

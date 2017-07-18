#define i2d(i) { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 }

static int estimate(const int i) {
    const int d[3] = i2d(i);
    int nhalodir[3] =  {
        d[0] != 0 ? 1 : XS,
        d[1] != 0 ? 1 : YS,
        d[2] != 0 ? 1 : ZS
    };

    int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
    int safety_factor = 2;
    return numberdensity * safety_factor * nhalocells;    
}

void ini_comm(const MPI_Comm cart, /**/ int rank[], int tags[]) {
    for (int i = 0; i < 27; ++i) {
        const int d[3] = i2d(i);
        tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
        int send_coor[3], ranks[3] = {m::coords[0], m::coords[1], m::coords[2]};
        for(int c = 0; c < 3; ++c) send_coor[c] = ranks[c] + d[c];
        l::m::Cart_rank(cart, send_coor, /**/ rank + i) ;
    }        
}

void ini_S(/**/ Send *s) {
    s->size_pin = new PinnedHostBuffer4<int>(27);
    for (int i = 0; i < 27; ++i) CC(cudaMalloc(&s->iidx_[i], sizeof(int) * estimate(i)));

    for (int i = 1; i < 27; ++i) alloc_pinned(i, 3 * estimate(i), /**/ &s->pp);
    CC(cudaMalloc(&s->pp.dp[0], sizeof(float) * 6 * estimate(0)));
    s->pp.hst[0] = NULL;

    CC(cudaMalloc(&s->size_dev, 27*sizeof(s->size_dev[0])));
    CC(cudaMalloc(&s->strt,     28*sizeof(s->strt[0])));

    CC(cudaMalloc(&s->iidx, SZ_PTR_ARR(s->iidx_)));
    CC(cudaMemcpy(s->iidx, s->iidx_, sizeof(s->iidx_), H2D));

    alloc_dev(/**/ &s->pp);

    if (global_ids) {
        for (int i = 1; i < 27; ++i) alloc_pinned(i, estimate(i), /**/ &s->ii);
        CC(cudaMalloc(&s->ii.dp[0], sizeof(int) * estimate(0)));
        s->ii.hst[0] = NULL;
        
        alloc_dev(/**/ &s->ii);
    }
}

void ini_R(const Send *s, /**/ Recv *r) {
    for (int i = 1; i < 27; ++i) alloc_pinned(i, 3 * estimate(i), /**/ &r->pp);
    r->pp.dp[0] = s->pp.dp[0];
    r->pp.hst[0] = NULL;

    CC(cudaMalloc(&r->strt, 28*sizeof(r->strt[0])));

    alloc_dev(/**/ &r->pp);

    if (global_ids) {
        for (int i = 1; i < 27; ++i) alloc_pinned(i, estimate(i), /**/ &r->ii);
        r->ii.dp[0] = s->ii.dp[0];
        r->ii.hst[0] = NULL;
        
        alloc_dev(/**/ &r->ii);
    }
}
#undef i2d

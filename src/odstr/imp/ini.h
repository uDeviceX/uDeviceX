namespace odstr {
namespace sub {

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

void ini_comm(/**/ int rank[], int tags[]) {
    for (int i = 0; i < 27; ++i) {
        const int d[3] = i2d(i);
        tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
        int send_coor[3], ranks[3] = {m::coords[0], m::coords[1], m::coords[2]};
        for(int c = 0; c < 3; ++c) send_coor[c] = ranks[c] + d[c];
        l::m::Cart_rank(l::m::cart, send_coor, /**/ rank + i) ;
    }        
}

void ini_S(/**/ Send *s) {
    int sz;
    dual::alloc(&s->size_pin, 27);
    for (int i = 0; i < 27; ++i)
        Dalloc(&s->iidx_[i], estimate(i));

    for (int i = 1; i < 27; ++i) alloc_pinned(i, 3 * estimate(i), /**/ &s->pp);
    Dalloc(&s->pp.dp[0], estimate(0));
    s->pp.hst[0] = NULL;

    Dalloc(&s->size_dev, 27);
    Dalloc(&s->strt,     28);

    sz = SZ_PTR_ARR(s->iidx_);
    Dalloc000(&s->iidx, sz);
    CC(d::Memcpy(s->iidx, s->iidx_, sizeof(s->iidx_), H2D));

    alloc_dev(/**/ &s->pp);
}

void ini_R(const Send *s, /**/ Recv *r) {
    for (int i = 1; i < 27; ++i) alloc_pinned(i, 3 * estimate(i), /**/ &r->pp);
    r->pp.dp[0] = s->pp.dp[0];
    r->pp.hst[0] = NULL;

    Dalloc(&r->strt, 28);

    alloc_dev(/**/ &r->pp);

}
#undef i2d

void ini_SRI(Pbufs<int> *sii, Pbufs<int> *rii) {
    for (int i = 1; i < 27; ++i) alloc_pinned(i, estimate(i), /**/ rii);
    for (int i = 1; i < 27; ++i) alloc_pinned(i, estimate(i), /**/ sii);
    
    Dalloc(&sii->dp[0], estimate(0));
    sii->hst[0] = NULL;
    rii->dp[0]  = sii->dp[0];
    rii->hst[0] = NULL;

    alloc_dev(/**/ sii);
    alloc_dev(/**/ rii);
}

} // sub
} // odstr

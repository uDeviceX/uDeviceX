enum {THR=128};

void oflocal(int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    float seed;
    seed = rnd_get(rnd);
    KL(flocaldev::apply_unroll,
       (ceiln((n), THR), THR),
       (n, cloud, start, seed, /**/ ff));
}

// try with textures
void flocal(int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    float seed;
    TBCloud tc;
    seed = rnd_get(rnd);
    
    setup0((float4*) cloud.pp, 2*n, &tc.pp);
    tc.cc = cloud.cc;
    
    KL(flocaldev::apply,
       (ceiln((n), THR), THR),
       (n, tc, start, seed, /**/ ff));

    destroy(&tc.pp);
}


enum {THR=128};

void flocal(int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    float seed;
    seed = rnd_get(rnd);
    KL(flocaldev::apply,
       (ceiln((n), THR), THR),
       (n, cloud, start, seed, /**/ ff));
}


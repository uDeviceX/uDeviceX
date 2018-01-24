
void flocal(int n, BCloud cloud, const int *start, const int *count, RNDunif *rnd, /**/ Force *ff) {
    float seed;
    seed = rnd_get(rnd);
    KL(flocaldev::apply,
       (k_cnf(n)),
       (n, cloud, start, count, seed, /**/ ff));
}


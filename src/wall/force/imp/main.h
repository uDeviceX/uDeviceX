enum {
    TPP = 3 /* # threads per particle */
};

void force(Wvel_v wv, Coords c, Cloud cloud, int n, RNDunif *rnd, Wa wa, /**/ Force *ff) {
    KL(dev::force,
       (k_cnf(TPP*n)),
       (wv, c, cloud, n, rnd_get(rnd), wa, /**/ (float*)ff));
}

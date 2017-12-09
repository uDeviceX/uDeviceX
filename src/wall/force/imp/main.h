void force(Wvel wv, Coords c, Cloud cloud, int n, rnd::KISS *rnd, Wa wa, /**/ Force *ff) {
    KL(dev::force,
       (k_cnf(3*n)),
       (wv.dev, c, cloud, n, rnd->get_float(), wa, /**/ (float*)ff));
}

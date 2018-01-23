enum {
    TPP = 3 /* # threads per particle */
};

void wall_force_apply(Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);
    
    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (wv, coordsv, cloud, n, rnd_get(rnd), wa, /**/ (float*)ff));
}

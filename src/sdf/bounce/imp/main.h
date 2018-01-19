void bounce_back(Wvel_v *wv, Coords *c, Sdf *sdf, int n, /**/ Particle *pp) {
    Sdf_v sdf_v;
    sdf_to_view(sdf, &sdf_v);
    KL(dev::bounce_back, (k_cnf(n)), (*wv, *c, sdf_v, n, /**/ pp));
}

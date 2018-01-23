void bounce_back(Wvel_v *wv, const Coords *c, Sdf *sdf, int n, /**/ Particle *pp) {
    Sdf_v sdf_v;
    Coords_v coordsv;
    sdf_to_view(sdf, &sdf_v);
    coords_get_view(c, &coordsv);
    KL(dev::bounce_back, (k_cnf(n)), (*wv, coordsv, sdf_v, n, /**/ pp));
}

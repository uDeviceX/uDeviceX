void bounce_back(float dt, const Wvel_v *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    Sdf_v sdf_v;
    Coords_v coordsv;
    sdf_to_view(sdf, &sdf_v);
    coords_get_view(c, &coordsv);
    KL(sdf_bb_dev::bounce_back, (k_cnf(n)), (dt, *wv, coordsv, sdf_v, n, /**/ pp));
}

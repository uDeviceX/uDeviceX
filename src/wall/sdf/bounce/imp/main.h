template <typename Wvel_v>
static void apply(float dt, Wvel_v wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    Sdf_v sdf_v;
    Coords_v coordsv;
    sdf_to_view(sdf, &sdf_v);
    coords_get_view(c, &coordsv);
    KL(sdf_bb_dev::bounce_back, (k_cnf(n)), (dt, wv, coordsv, sdf_v, n, /**/ pp));
}
    
void bounce_back(float dt, const WvelStep *w, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    switch (wvel_get_type(w)) {
    case WALL_VEL_V_CSTE: {
        WvelCste_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    case WALL_VEL_V_SHEAR: {
        WvelShear_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    case WALL_VEL_V_HS: {
        WvelHS_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    };
}

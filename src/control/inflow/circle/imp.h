static bool is_inside_subdomain(Coords c, float3 r) {
    bool inside = true;
    inside &= (xlo(c) <= r.x) && (r.x < xhi(c));
    inside &= (ylo(c) <= r.y) && (r.y < yhi(c));
    inside &= (zlo(c) <= r.z) && (r.z < zhi(c));
    return inside;
}

void ini_params_circle(Coords c, float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i) {
    enum {X, Y, Z};
    circle::Params *pp;
    circle::VParams *vpp;

    if (is_inside_subdomain(c, o)) {    
        pp  = &i->p.circle;
        vpp = &i->vp.circle;

        global2local(c, o, /**/ &o);
    
        pp->o = o;
        pp->R = R;
        pp->H = H;

        if (R > XS/2 || R > YS/2 || R > ZS/2)
            ERR("radius should be smaller");
    
        // TODO
        pp->th0 = 0;
        pp->dth = 2 * M_PI;
    
        vpp->u = u;
        vpp->poiseuille = poiseuille;

        i->t = TYPE_CIRCLE;
    } else {
        i->t = TYPE_NONE;
    }
}

void ini_params_circle(Coords c, float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i) {
    enum {X, Y, Z};
    circle::Params *pp;
    circle::VParams *vpp;

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
}

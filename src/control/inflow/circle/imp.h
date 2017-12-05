void ini_params_circle(float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i) {
    enum {X, Y, Z};
    circle::Params *pp;
    circle::VParams *vpp;

    pp  = &i->p.circle;
    vpp = &i->vp.circle;

    float3 mi;
    mi.x = (m::coords[X] + 0.5) * XS;
    mi.y = (m::coords[Y] + 0.5) * YS;
    mi.z = (m::coords[Z] + 0.5) * ZS;

    // shift to local referential
    axpy(-1.0, &mi, /**/ &o);
    
    pp->o = o;
    pp->R = R;
    pp->H = H;
    // TODO
    pp->th0 = 0;
    pp->dth = 2 * M_PI;
    
    vpp->u = u;
    vpp->poiseuille = poiseuille;

    i->t = TYPE_CIRCLE;
}

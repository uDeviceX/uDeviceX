void ini_params_circle(float3 o, float R, float H, float u, bool poiseuille,
                       /**/ Inflow *i) {
    enum {X, Y, Z};
    circle::Params *pp;
    circle::VParams *vpp;

    pp  = &i->p.circle;
    vpp = &i->vp.circle;
    
    vpp->u = u;
    vpp->poiseuille = poiseuille;

    i->t = TYPE_CIRCLE;
}

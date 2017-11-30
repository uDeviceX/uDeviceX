void ini_params_plate(float3 o, float3 a, float3 b,
                      float3 u, bool upoiseuille, bool vpoiseuille,
                      /**/ Inflow *i) {
    plate::Params *pp;
    plate::VParams *vpp;

    pp = &i->p.plate;
    vpp = &i->vp.plate;
    
    pp->o = o;
    pp->a = a;
    pp->b = b;

    vpp->u = u;
    vpp->upoiseuille = upoiseuille;
    vpp->vpoiseuille = vpoiseuille;
}

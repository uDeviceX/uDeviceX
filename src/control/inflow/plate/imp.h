static float crop(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float3 crop(float3 r, float3 lo, float3 hi) {
    r.x = crop(r.x, lo.x, hi.x);
    r.y = crop(r.y, lo.y, hi.y);
    r.z = crop(r.z, lo.z, hi.z);
    return r;
}

void ini_params_plate(float3 o, int dir, float L1, float L2,
                      float3 u, bool upoiseuille, bool vpoiseuille,
                      /**/ Inflow *i) {
    enum {X, Y, Z};
    plate::Params *pp;
    plate::VParams *vpp;
    float3 a, b, lo, hi;
    float3 co, ca, cb; // corners of the local plane : co (origin), ca, cb
    int mi[3], L[3] = {XS, YS, ZS};
    for (int c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    lo = make_float3(-XS/2, -YS/2, -ZS/2);
    hi = make_float3( XS/2,  YS/2,  ZS/2);
    
    pp = &i->p.plate;
    vpp = &i->vp.plate;

    a = b = make_float3(0, 0, 0);
    
    switch(dir) {
    case X:
        a.y = L1;
        b.z = L2;
        break;
    case Y:
        a.x = L1;
        b.z = L2;
        break;
    case Z:
        a.x = L1;
        b.y = L2;
        break;
    default:
        ERR("wrong direction");
        break;
    };

    // shift to local coordinates
    o.x -= mi[X];
    o.y -= mi[Y];
    o.z -= mi[Z];

    co = ca = cb = o;
    add(&a, /**/ &ca);
    add(&b, /**/ &cb);

    co = crop(co, lo, hi);
    ca = crop(ca, lo, hi);
    cb = crop(cb, lo, hi);
    
    pp->o = co;
    diff(&ca, &co, /**/ &pp->a);
    diff(&cb, &co, /**/ &pp->b);

    printf("%g %g %g\n", co.x, co.y, co.z);
    printf("%g %g %g\n", pp->a.x, pp->a.y, pp->a.z);
    printf("%g %g %g\n", pp->b.x, pp->b.y, pp->b.z);
    
    vpp->u = u;
    vpp->upoiseuille = upoiseuille;
    vpp->vpoiseuille = vpoiseuille;

    i->t = TYPE_PLATE;
}

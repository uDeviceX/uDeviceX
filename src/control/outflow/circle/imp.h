void ini_params_circle(float3 c, float R, Outflow *o) {
    enum {X, Y, Z};
    
    circle::Params p;
    float3 mi;
    mi.x = (m::coords[X] + 0.5) * XS;
    mi.y = (m::coords[Y] + 0.5) * YS;
    mi.z = (m::coords[Z] + 0.5) * ZS;

    p.Rsq = R*R;
    
    // shift to local referential
    axpy(-1.0, &mi, /**/ &c);

    p.c = c;

    o->params.circle = p;
    o->type = TYPE_CIRCLE;
}

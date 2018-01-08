void ini_params_circle(Coords coords, float3 c, float R, Outflow *o) {
    enum {X, Y, Z};
    ParamsCircle p;
    p.Rsq = R*R;
    
    // shift to local referential
    global2local(coords, c, /**/ &p.c);

    o->params.circle = p;
    o->type = TYPE_CIRCLE;
}

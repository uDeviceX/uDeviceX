void ini_params_circle(float R, Outflow *o) {
    circle::Params p;
    p.Rsq = R*R;
    // TODO
    p.c = make_float3(0, 0, 0);

    o->params.circle = p;
    o->type = TYPE_CIRCLE;
}

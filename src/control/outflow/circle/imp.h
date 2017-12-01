void ini_params_circle(float R, Outflow *o) {
    circle::Params p;
    p.Rsq = R*R;

    o->params.circle = p;
    o->type = TYPE_CIRCLE;
}

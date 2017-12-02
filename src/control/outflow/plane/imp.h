void ini_params_plane(int dir, float r0, Outflow *o) {
    enum {X, Y, Z};
    plane::Params p = {0, 0, 0, r0};

    switch (dir) {
    case X:
        p.a = 1;
        break;
    case Y:
        p.b = 1;
        break;
    case Z:
        p.c = 1;
        break;
    default:
        ERR("undefined direction %d\n", dir);
        break;
    };

    o->params.plane = p;
    o->type = TYPE_PLANE;
}

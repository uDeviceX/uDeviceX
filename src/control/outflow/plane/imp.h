void ini_params_plane(int dir, float r0, Outflow *o) {
    enum {X, Y, Z};
    plane::Params p = {0, 0, 0, r0};    

    switch (dir) {
    case X:
        p.a = 1;
        p.d -= (m::coords[X] + 0.5) * XS;
        break;
    case Y:
        p.b = 1;
        p.d -= (m::coords[Y] + 0.5) * YS;
        break;
    case Z:
        p.c = 1;
        p.d -= (m::coords[Z] + 0.5) * ZS;
        break;
    default:
        ERR("undefined direction %d\n", dir);
        break;
    };

    
    
    o->params.plane = p;
    o->type = TYPE_PLANE;
}

void ini_params_plane(Coords c, int dir, float r0, Outflow *o) {
    enum {X, Y, Z};
    ParamsPlane p = {0, 0, 0, 0};    

    switch (dir) {
    case X:
        p.a = 1;
        p.d = xg2xl(c, r0);
        break;
    case Y:
        p.b = 1;
        p.d = yg2yl(c, r0);
        break;
    case Z:
        p.c = 1;
        p.d = zg2zl(c, r0);
        break;
    default:
        ERR("undefined direction %d\n", dir);
        break;
    };    
    
    o->params.plane = p;
    o->type = TYPE_PLANE;
}

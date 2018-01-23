static void color(const Coords *coords, Particle *pp, int n, /**/ int *cc) {
    enum {X, Y, Z};
    int lx, ly, lz; /* domain */
    int i, r, g;
    bool inside;
    float x, y, z, rad;
    float x0, y0, z0;
    Particle p;
    rad = BANGLADESH_R;

    lx = xdomain(coords);
    ly = ydomain(coords);
    lz = zdomain(coords);
    
    x0 = 0.5*lx; y0 = 0.5*ly; z0 = 0.5*lz;
    enum {R = RED_COLOR, G = BLUE_COLOR};
    for (i = r = g = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X]; y = p.r[Y]; z = p.r[Z];
        x = xl2xg(coords, x);
        y = yl2yg(coords, y);
        z = zl2zg(coords, z);
        x -= x0; y -= y0; z -= z0;
        inside = x*x + y*y + z*z < rad*rad;
        if (inside) {cc[i] = R; r++;}
        else        {cc[i] = G; g++;}
    }
    msg_print("color scheme: Bangladesh");
    msg_print("red/green : %d/%d", r, g);
}

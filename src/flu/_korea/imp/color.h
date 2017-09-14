static int min (int x, int y)        { return x < y ? x : y; };
static int min3(int x, int y, int z) { return min(x, min(y, z)); }
static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X, Y, Z};
    int lx, ly, lz; /* domain */
    int i, r, g;
    bool inside;
    float x, y, z, rad;
    float x0, y0, z0;
    Particle p;
    lx = m::lx(); ly = m::ly(); lz = m::lz();
    x0 = 0.5*lx; y0 = 0.5*ly; z0 = 0.5*lz;
    enum {R = RED_COLOR, G = BLUE_COLOR};
    rad = 0.25*min3(lx, ly, lz);
    for (i = r = g = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X]; y = p.r[Y]; z = p.r[Z];
        x = m::x2g(x); y = m::y2g(y); z = m::z2g(z);
        x -= x0; y -= y0; z -= z0;
        inside = x*x + y*y + z*z < rad*rad;
        if (inside) {cc[i] = R; r++;}
        else        {cc[i] = G; g++;}
    }
    MSG("color scheme: Korea");
    MSG("red/green : %d/%d", r, g);
}

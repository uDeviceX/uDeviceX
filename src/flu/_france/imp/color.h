static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X};
    int lx; /* domain */
    int i;
    float x;
    Particle p;
    lx = m::lx();

    enum {B = BLUE_COLOR, W = RED_COLOR, R = RED_COLOR}; /* white is red in France */
    for (i = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X];
        x += 0.5 * lx;
        x *= 3;
        if      (x <     lx) cc[i] = B;
        else if (x < 2 * lx) cc[i] = W;
        else                 cc[i] = R;
    }
}

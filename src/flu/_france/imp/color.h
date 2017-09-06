static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X, Y};
    int ly; /* domain */
    int i, b, w, r;
    float y;
    Particle p;
    ly = m::ly();

    enum {B = BLUE_COLOR, W = RED_COLOR, R = RED_COLOR}; /* white is red in France */
    for (i = b = w = r = 0; i < n; i++) {
        p = pp[i];
        y = p.r[Y];
        y += 0.5 * ly;
        y *= 3;
        if      (y <     ly) {cc[i] = B; b++;}
        else if (y < 2 * ly) {cc[i] = W; w++;}
        else                 {cc[i] = R; r++;}
    }
    MSG("color scheme: France");
    MSG("blue/white/red : %d/%d/%d", b, w, r);
}

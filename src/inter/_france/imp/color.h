static void color(const Coords *coords, Particle *pp, int n, /**/ int *cc) {
    enum {X, Y};
    int ly; /* domain */
    int i, b, w, r;
    float y;
    Particle p;
    ly = ydomain(coords);

    enum {B = BLUE_COLOR, W = RED_COLOR, R = RED_COLOR}; /* white is red in France */
    for (i = b = w = r = 0; i < n; i++) {
        p = pp[i];
        y = p.r[Y];
        y = yl2yg(coords, y); /* to global */

        y *= 2;
        if      (y <     ly) {cc[i] = B; b++;}
        else if (y < 2 * ly) {cc[i] = W; w++;}
        else                 {cc[i] = R; r++;}
    }
    msg_print("color scheme: France");
    msg_print("blue/white/red : %d/%d/%d", b, w, r);
}

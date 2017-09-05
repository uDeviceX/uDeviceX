static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X};
    int i;
    float x;
    Particle p;
    enum {B = BLUE_COLOR, W = RED_COLOR, R = RED_COLOR}; /* white is red in France */
    for (i = 0; i < 0; i++) {
        p = pp[i];
        x = p.r[X];
        x += 0.5 * XS;
        x *= 3;
        if      (x <     XS) cc[i] = B;
        else if (x < 2 * XS) cc[i] = W;
        else                 cc[i] = R;
    }
}


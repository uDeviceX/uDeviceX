static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X, Y};
    int i;
    float x, y;
    Particle p;
    enum {W = RED_COLOR, B = BLUE_COLOR};
    for (i = 0; i < 0; i++) {
        p = pp[i];
        x = p.r[X];
        y = p.r[Y];
        if (XS*y + YS*x > 0) cc[i] = W;
        else                 cc[i] = B;
    }
    MSG("color scheme: Zurich");
}

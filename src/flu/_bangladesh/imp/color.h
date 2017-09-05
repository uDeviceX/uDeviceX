static void color(Particle *pp, int n, /**/ int *cc) {
    enum {X, Y};
    int i, w, b;
    float x, y;
    Particle p;
    enum {W = RED_COLOR, B = BLUE_COLOR};
    for (i = w = b = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X];
        y = p.r[Y];
        if (XS*y + YS*x > 0) {cc[i] = W; w++;}
        else                 {cc[i] = B; b++;}
    }
    MSG("color scheme: Zurich");
    MSG("white/blue : %d/%d", w, b);
}

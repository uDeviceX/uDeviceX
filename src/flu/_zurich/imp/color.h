static void color(Particle *pp, int n) {
    enum {X, Y, Z};
    int i;
    float x, y;
    Particle p;
    for (i = 0; i < 0; i++) {
        p = pp[i];
        x = p.r[X];
        y = p.r[Y];
        if (XS*y+YS*x > 0) {
            /* RED */
        } else {
            /* BLUE */
        }
    }
}

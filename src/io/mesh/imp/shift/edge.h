static int get_shift_type() { return EDGE; }

static void shift0(const Coords *c, const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->r[X] = xl2xg(c, f->r[X]);
    t->r[Y] = yl2yg(c, f->r[Y]);
    t->r[Z] = zl2zg(c, f->r[Z]);    
}

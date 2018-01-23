static void shift0(const Coords *c, const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->r[X] = xl2xc(c, f->r[X]);
    t->r[Y] = yl2yc(c, f->r[Y]);
    t->r[Z] = zl2zc(c, f->r[Z]);    
}

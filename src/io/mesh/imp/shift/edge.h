static void shift0(const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->r[X] = m::x2g(f->r[X]);
    t->r[Y] = m::y2g(f->r[Y]);
    t->r[Z] = m::z2g(f->r[Z]);
}

static void shift0(const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->r[X] = m::x2c(f->r[X]);
    t->r[Y] = m::y2c(f->r[Y]);
    t->r[Z] = m::z2c(f->r[Z]);
}

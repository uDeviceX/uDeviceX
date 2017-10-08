void vel(const int *cc, int color, int n, /**/ Particle *pp) {
    dev::Map m;
    m.cc = cc; m.color = color;
    vel0(m, n, /**/ pp);
}

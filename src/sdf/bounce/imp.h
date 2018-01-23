struct Sdf;
struct Wvel_v;
struct Coords;
struct Particle;

void bounce_back(Wvel_v *wv, const Coords *c, Sdf *sdf, int n, /**/ Particle *pp);

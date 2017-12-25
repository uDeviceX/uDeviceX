struct Sdf;
struct Wvel_v;
struct Coords;
struct Particle;

void bounce_back(Wvel_v *wv, Coords *c, Sdf *sdf, int n, /**/ Particle *pp);

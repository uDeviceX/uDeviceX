struct Sdf;
struct Wvel_v;
struct Coords;
struct Particle;

void bounce_back(float dt0, const Wvel_v *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);

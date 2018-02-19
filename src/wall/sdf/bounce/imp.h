struct Sdf;
struct WvelStep;
struct Coords;
struct Particle;

void bounce_back(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);

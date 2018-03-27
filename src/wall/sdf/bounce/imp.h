struct Sdf;
struct WvelStep;
struct Coords;
struct Particle;

// tag::int[]
void bounce_back(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);
// end::int[]


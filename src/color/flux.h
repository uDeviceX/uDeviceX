struct int3;
struct Particle;
struct Coords;

void color_linear_flux(const Coords*, int3 L, int dir, int color, int n, const Particle *pp, /**/
                       int *cc);

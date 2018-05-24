struct int3;
struct Particle;
struct Coords;

struct Rnd;

void color_linear_flux(const Coords*, int3 L, int dir, int color, int n, const Particle *pp, /**/int *cc);

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);
void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);

void tracers_ini(const int n);

void tracers_fin();

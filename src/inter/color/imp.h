struct GenColor;
struct Particle;
struct Coords;

void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);

void inter_color_set_drop(float R, GenColor*);
void inter_color_set_uniform(GenColor*);

void inter_color_apply(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);

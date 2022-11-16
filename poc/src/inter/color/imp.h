struct GenColor;
struct Particle;
struct Coords;
struct Config;

// tag::mem[]
void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);
// end::mem[]

// tag::set[]
void inter_color_set_drop(float R, GenColor*); // <1>
void inter_color_set_uniform(GenColor*); // <2>
// end::set[]

// tag::cfg[]
void inter_color_set_conf(const Config*, GenColor*);
// end::cfg[]

// tag::int[]
void inter_color_apply_hst(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
void inter_color_apply_dev(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
// end::int[]


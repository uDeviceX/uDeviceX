struct DCont;
struct DContMap;

struct Config;
struct Coords;

// tag::mem[]
void den_ini(int maxp, /**/ DCont**);
void den_map_ini(DContMap**);

void den_fin(DCont*);
void den_map_fin(DContMap*);
// end::mem[]

// tag::set[]
void den_map_set_none(const Coords*, DContMap*);
void den_map_set_circle(const Coords*, float R, DContMap*);
// end::set[]

// tag::cfg[]
void den_map_set_conf(const Config*, const Coords*, DContMap*);
// end::cfg[]

// tag::int[]
void den_reset(int n, /**/ DCont*d); // <1>
void den_filter_particles(int maxdensity, const DContMap*, const int *starts, const int *counts, /**/ DCont*); // <2>
void den_download_ndead(DCont*); // <3>
// end::int[]

// tag::get[]
int* den_get_deathlist(DCont*); // <1>
int  den_get_ndead(DCont*); // <2>
// end::get[]

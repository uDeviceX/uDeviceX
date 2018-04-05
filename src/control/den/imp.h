struct DCont;
struct DContMap;

struct Coords;

void den_ini(int maxp, /**/ DCont**);
void den_map_ini(DContMap**);

void den_fin(DCont*);
void den_map_fin(DContMap*);

void den_map_set_none(const Coords*, DContMap*);
void den_map_set_circle(const Coords*, float R, DContMap*);

void den_reset(int n, /**/ DCont *d);
void den_filter_particles(int maxdensity, const DContMap *m, const int *starts, const int *counts, /**/ DCont *d);
void den_download_ndead(DCont *d);

int* den_get_deathlist(DCont *d);
int  den_get_ndead(DCont *d);

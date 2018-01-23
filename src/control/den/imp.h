struct DCont;
struct DContMap;

void den_ini(int maxp, /**/ DCont **d0);
void den_ini_map_none(const Coords *c, DContMap **m0);
void den_ini_map_circle(const Coords *c, float R, DContMap **m0);

void den_fin(DCont *d);
void den_fin_map(DContMap *m);

void den_reset(int n, /**/ DCont *d);
void den_filter_particles(const DContMap *m, const int *starts, const int *counts, /**/ DCont *d);
void den_download_ndead(DCont *d);

int* den_get_deathlist(DCont *d);
int  den_get_ndead(DCont *d);

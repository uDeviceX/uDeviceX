struct DCont;
struct DContMap;

void ini(int maxp, /**/ DCont **d0);
void ini(Coords coords, DContMap **m0);

void fin(DCont *d);
void fin(DContMap *m);

void reset(int n, /**/ DCont *d);
void filter_particles(const DContMap *m, const int *starts, const int *counts, /**/ DCont *d);
void download_ndead(DCont *d);

int* get_deathlist(DCont *d);
int  get_ndead(DCont *d);

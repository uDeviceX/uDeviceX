struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

void adj_ini(int md, int nt, int nv, const int4 *faces, /**/ Adj*);
void adj_fin(Adj*);
int  adj_get_map(int md, int nv, int i, const Adj*, /**/ AdjMap *m);
void adj_get_anti(int md, int nv, const Adj*, /**/ int *anti);

void adj_view_ini(Adj*, Adj_v**);
void adj_view_fin(Adj_v*);

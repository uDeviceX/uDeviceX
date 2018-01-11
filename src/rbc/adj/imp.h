struct Adj;
struct AdjMap;

void adj_ini(int md, int nt, int nv, int4 *faces, /**/ Adj*);
void adj_fin(Adj*);
int adj_get_map(int md, int nv, int i, const Adj*, /**/ AdjMap *m);

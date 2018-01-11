namespace adj {
struct Adj;
struct AdjMap; /* see type.h */
void adj_ini(int md, int nt, int nv, int4 *faces, /**/ Adj*);
void adj_fin(Adj*);
int hst(int md, int nv, int i, const Adj*, /**/ AdjMap *m);

} /* namespace */

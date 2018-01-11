namespace adj {
struct Adj;
struct AdjMap; /* see type.h */
void ini(int md, int nt, int nv, int4 *faces, /**/ Adj*);
void fin(Adj*);
int hst(int md, int nv, int i, Adj*, /**/ AdjMap *m);

} /* namespace */

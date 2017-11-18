namespace adj {
struct Hst;
struct Map; /* see type.h */
void ini(int md, int nt, int nv, int4 *faces, /**/ Hst*);
void fin(Hst*);
int hst(int md, int nv, int i, Hst*, /**/ Map *m);

} /* namespace */

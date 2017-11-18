namespace rbc { namespace adj {
struct Hst;
struct Map; /* see type.h */
void ini0(int md, int nt, int nv, int4 *faces, /**/ int *adj0, int *adj1);

int hst0(int md, int nv, int i, int *adj0, int *adj1, /**/ Map *m);

void ini(int md, int nt, int nv, int4 *faces, /**/ Hst*);
void fin(Hst*);
int hst(int md, int nv, int i, Hst*, /**/ Map *m);

}} /* namespace */

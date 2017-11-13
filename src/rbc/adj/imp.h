namespace rbc { namespace adj {
struct Map; /* see type.h */
void ini(int md, int nt, int nv, int4 *faces, /**/ int *adj0, int *adj1);
int map(int md, int nv, int i, int *adj0, int *adj1, /**/ Map *m);
}} /* namespace */

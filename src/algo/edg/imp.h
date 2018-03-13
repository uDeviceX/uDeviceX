// tag::interface[]
void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val, /**/ int *hx,       int *hy); // <1>
int  edg_get(int md, int i, int j,         const int *hx, const int *hy); // <2>
// end::interface

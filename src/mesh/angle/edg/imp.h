struct Edg;

/* val0: initial value */
void e_ini(int md, int nv, int val0, /**/ Edg**);
void e_fin(Edg*);

void e_set(Edg*, int i, int j, int val);
int  e_get(Edg*, int i, int j);

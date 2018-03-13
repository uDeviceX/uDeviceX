struct Edg;

void e_ini(int md, int nv, /**/ Edg**);
void e_fin(Edg*);

void e_set(Edg*, int i, int j, int val);
int  e_get(Edg*, int i, int j);

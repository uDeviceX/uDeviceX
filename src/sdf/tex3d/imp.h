struct Tex3d;
void tex3d_ini(Tex3d**, int x, int y, int z);
void tex3d_fin(Tex3d*);
void tex3d_copy(int x, int y, int z, float *D, /**/ Tex3d*);

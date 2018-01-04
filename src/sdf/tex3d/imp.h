struct Tex3d;
struct Tex3d_v;
struct Array3d;
void tex3d_ini(Tex3d**);
void tex3d_fin(Tex3d*);
void tex3d_copy(Array3d*, /**/ Tex3d*);
void tex3d_to_view(Tex3d*, /**/ Tex3d_v*);

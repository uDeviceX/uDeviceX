struct int4;
float mesh_volume(int nt, const int4 *tt, const float *vv);
void mesh_center_of_mass(int nt, const int4 *tt, const float *vv, /**/ float *com);
void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density, /**/ float *I);

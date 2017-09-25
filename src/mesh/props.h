namespace mesh
{
float volume(int nt, const int *tt, const float *vv);
void center_of_mass(int nt, const int *tt, const float *vv, /**/ float *com);
void inertia_tensor(int nt, const int *tt, const float *vv, const float *com, const float density, /**/ float *I);    
}

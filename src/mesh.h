namespace mesh
{
    void center_of_mass(const Mesh mesh, /**/ float *com);
    void inertia_tensor(const Mesh mesh, const float *com, const float density, /**/ float *I);    
}

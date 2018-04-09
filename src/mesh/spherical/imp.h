struct MeshSpherical;
struct Vectors;

// tag::interface[]
void mesh_spherical_ini(int nv, /**/ MeshSpherical**);
void mesh_spherical_fin(MeshSpherical*);
void  mesh_spherical_apply(MeshSpherical*, int nm, Vectors*, /**/
                           double *r, double *theta, double *phi);
// end::interface[]

struct MeshSpherical;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_spherical_ini(MeshRead*, /**/ MeshSpherical**); // <1>
void mesh_spherical_fin(MeshSpherical*);
void  mesh_spherical_apply(MeshSpherical*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

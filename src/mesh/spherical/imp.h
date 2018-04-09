struct MeshSpherical;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_angle_ini(MeshRead*, /**/ MeshSpherical**); // <1>
void mesh_angle_fin(MeshSpherical*);
void  mesh_angle_apply(MeshSpherical*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

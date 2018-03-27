struct MeshVertArea;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_tri_vert_ini(MeshRead*, /**/ MeshVertArea**); // <1>
void mesh_tri_vert_fin(MeshVertArea*);

void  mesh_tri_vert_apply(MeshVertArea*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

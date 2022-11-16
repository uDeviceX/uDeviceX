struct MeshVertArea;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_vert_area_ini(MeshRead*, /**/ MeshVertArea**); // <1>
void mesh_vert_area_fin(MeshVertArea*);

void  mesh_vert_area_apply(MeshVertArea*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

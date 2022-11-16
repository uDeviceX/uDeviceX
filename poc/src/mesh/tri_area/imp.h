struct MeshTriArea;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_tri_area_ini(MeshRead*, /**/ MeshTriArea**); // <1>
void mesh_tri_area_fin(MeshTriArea*);

void  mesh_tri_area_apply(MeshTriArea*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

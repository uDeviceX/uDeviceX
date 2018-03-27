struct MeshArea;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_area_ini(MeshRead*, /**/ MeshArea**); // <1>
void mesh_area_fin(MeshArea*);

double mesh_area_apply0(MeshArea*, Vectors*);                     // <2>
void  mesh_area_apply(MeshArea*, int nm, Vectors*, /**/ double*); // <3>
// end::interface[]

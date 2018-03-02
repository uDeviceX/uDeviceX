struct MeshArea;
struct Positions;
struct MeshRead;

// tag::interface[]
void mesh_area_ini(MeshRead*, /**/ MeshArea**); // <1>
void mesh_area_fin(MeshArea*);

double mesh_area_apply0(MeshArea*, Positions*);                      // <2>
void  mesh_area_apply(MeshArea*, int nm, Positions*, /**/ double*); // <3>
// end::interface[]

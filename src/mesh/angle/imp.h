struct MeshTriArea;
struct Positions;
struct MeshRead;

// tag::interface[]
void mesh_tri_area_ini(MeshRead*, /**/ MeshTriArea**); // <1>
void mesh_tri_area_fin(MeshTriArea*);

void  mesh_tri_area_apply(MeshTriArea*, int nm, Positions*, /**/ double*); // <2>
// end::interface[]

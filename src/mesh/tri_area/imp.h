struct MeshTriArea;
struct Positions;
struct MeshRead;

// tag::interface[]
void mesh_tri_area_ini(MeshRead*, /**/ MeshTriArea**); // <1>
void mesh_tri_area_fin(MeshTriArea*);

double mesh_tri_area_apply0(MeshTriArea*, Positions*);                      // <2>
void  mesh_tri_area_apply(MeshTriArea*, int nm, Positions*, /**/ double*); // <3>
// end::interface[]

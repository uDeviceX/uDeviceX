struct MeshScatter;
struct MeshRead;
struct Scalars;

// tag::interface[]
void mesh_scatter_ini(MeshRead*, MeshScatter**); // <1>
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
// end::interface[]

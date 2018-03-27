struct MeshScatter;

// tag::interface[]
void mesh_scatter_ini(MeshRead*, MeshScatter**);
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*);
// end::interface[]

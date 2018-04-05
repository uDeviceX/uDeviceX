struct MeshEdgLen;
struct MeshRead;
struct Vectors;

// tag::interface[]
void mesh_edg_len_ini(MeshRead*, MeshEdgLen**);
void mesh_edg_len_fin(MeshEdgLen*);
void mesh_edg_len_apply(MeshEdgLen*, int nm, Vectors*, /**/ double*);
// end::interface[]

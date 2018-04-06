struct MeshEngKantor;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_eng_kantor_ini(MeshRead*, int max_nm, /**/ MeshEngKantor**); // <1>
void mesh_eng_kantor_fin(MeshEngKantor*);

void  mesh_eng_kantor_apply(MeshEngKantor*, int nm, Vectors*, double kb, /**/ double*); // <2>
// end::interface[]

struct MeshEngJulicher;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_eng_julicher_ini(MeshRead*, /**/ MeshEngJulicher**); // <1>
void mesh_eng_julicher_fin(MeshEngJulicher*);

void  mesh_eng_julicher_apply(MeshEngJulicher*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

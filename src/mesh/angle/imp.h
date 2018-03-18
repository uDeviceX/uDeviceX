struct MeshAngle;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);

void  mesh_angle_apply(MeshAngle*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]

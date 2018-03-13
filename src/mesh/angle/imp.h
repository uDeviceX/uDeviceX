struct MeshAngle;
struct Positions;
struct MeshRead;

// tag::interface[]
void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);

void  mesh_angle_apply(MeshAngle*, int nm, Positions*, /**/ double*); // <2>
// end::interface[]

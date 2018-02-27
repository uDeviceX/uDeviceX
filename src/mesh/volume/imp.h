struct MeshVolume;
struct Positions;
struct MeshRead;

// tag::interface[]
void mesh_volume_ini(MeshRead*, /**/ MeshVolume**); // <1>
void mesh_volume_fin(MeshVolume*);

double mesh_volume_apply0(MeshVolume*, Positions*);                      // <2>
void  mesh_volume_apply(MeshVolume*, int nm, Positions*, /**/ double**); // <3>
// end::interface[]

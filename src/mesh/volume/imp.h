struct MeshVolume;
struct Positions;
struct MeshRead;

void mesh_volume_ini(MeshRead*, /**/ MeshVolume**);
void mesh_volume_fin(MeshVolume*);

void  mesh_volume_apply(MeshVolume*, int nm, Positions*, /**/ float**);
float mesh_volume_apply0(MeshVolume*, Positions*);
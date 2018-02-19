struct DiagMesh;
struct MeshRead;

void diag_mesh_ini(const char *path, MeshRead*, DiagMesh**);
void diag_mesh_fin(DiagMesh*);
void diag_mesh_apply(DiagMesh*, MPI_Comm, float time, int nc, Particle*);

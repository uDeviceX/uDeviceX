struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct OffRead;

void mesh_write_ini(const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_off(OffRead*, const char *directory, /**/ MeshWrite**);
void mesh_write_dump(MeshWrite*, MPI_Comm, const Coords*, int nc, const Particle*, int id);
void mesh_write_fin(MeshWrite*);

struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

void mesh_write_ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);

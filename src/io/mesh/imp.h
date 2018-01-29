struct Coords;
struct Particle;
struct int4;

void mesh_write_rbc(MPI_Comm, const Coords*, const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);
void mesh_write_rig(MPI_Comm, const Coords*, const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);

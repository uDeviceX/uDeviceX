namespace io { namespace mesh {
void rbc(MPI_Comm cart, const Coords *coords, const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);
void rig(MPI_Comm cart, const Coords *coords, const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);
}}

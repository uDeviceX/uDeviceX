namespace io { namespace mesh {
void main(const Particle*, const int4 *faces, int nc, int nv, int nt, const char*);
void rbc(const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);
void rig(const Particle*,  const int4 *faces, int nc, int nv, int nt, int id);
}}

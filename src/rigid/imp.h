namespace rig {

void reinit_ft(const int nsolid, /**/ Solid *ss);
void update(int n, const Force *ff, const float *rr0, int nsolid, /**/ Particle *pp, Solid *ss);
void generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp);
void update_mesh(int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp);

} // rig

void rig_reinit_ft(const int nsolid, /**/ Solid *ss);
void rig_update(int n, const Force *ff, const float *rr0, int nsolid, /**/ Particle *pp, Solid *ss);
void rig_generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp);
void rig_update_mesh(int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp);

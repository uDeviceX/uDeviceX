void rig_reinit_ft(const int nsolid, /**/ Solid *ss);
void rig_update(float dt, int n, const Force *ff, const float *rr0, int nsolid, /**/ Particle *pp, Solid *ss);
void rig_generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp);
void rig_update_mesh(float dt, int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp);

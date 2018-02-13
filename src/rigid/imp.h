struct RigPinInfo;
struct int3;
struct Config;

void rig_ini_pininfo(RigPinInfo **);
void rig_fin_pininfo(RigPinInfo *);
void rig_set_pininfo(int3 com, int3 axis, RigPinInfo *);
void rig_set_pininfo_conf(const Config *cfg, RigPinInfo *);

struct Particle;
struct Solid;
struct Force;

void rig_reinit_ft(const int nsolid, /**/ Solid *ss);
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int nsolid, /**/ Particle *pp, Solid *ss);
void rig_generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp);
void rig_update_mesh(float dt, int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp);

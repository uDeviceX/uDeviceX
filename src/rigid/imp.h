struct RigPinInfo;
struct int3;
struct Config;

// tag::enum[]
enum {NOT_PERIODIC = -1};
// end::enum[]

// tag::pin[]
void rig_ini_pininfo(RigPinInfo **); // <1>
void rig_fin_pininfo(RigPinInfo *); // <2>
void rig_set_pininfo(int3 com, int3 axis, RigPinInfo *); // <3>
void rig_set_pdir(int pdir, RigPinInfo *); // <4>
void rig_set_pininfo_conf(const Config *cfg, RigPinInfo *); // <5>

int rig_get_pdir(const RigPinInfo *); // <6>
// end::pin[]

struct Particle;
struct Solid;
struct Force;

// tag::upd[]
void rig_reinit_ft(const int ns, /**/ Solid *ss); // <1>
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Solid *ss); // <2>
void rig_generate(int ns, const Solid *ss, int nps, const float *rr0, /**/ Particle *pp); // <3>
void rig_update_mesh(float dt, int ns, const Solid *ss, int nv, const float *vv, /**/ Particle *pp); // <4>
// end::upd[]


struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;
struct RbcParams;

// tag::mem[]
void rbc_bending_ini(const MeshRead *cell, RbcForce**);
void rbc_bending_fin(RbcForce*);
// end::mem[]

// tag::set[]
void rbc_bending_set_stressful(int nt, float totArea, /**/ RbcForce*); // <1>
void rbc_bending_set_stressfree(const char *fname, /**/ RbcForce*);    // <2>

void rbc_bending_set_rnd0(RbcForce *f);           // <3>
void rbc_bending_set_rnd1(int seed, RbcForce *f); // <4>
// end::set[]

// tag::cnf[]
void rbc_bending_set_conf(const MeshRead *cell, const Config *cfg, const char *name, RbcForce *f);
// end::cnf[]

// tag::apply[]
void rbc_bending_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*); // <1>
void rbc_bending_stat(/**/ float *pArea, float *pVolume); // <2>
// end::apply[]
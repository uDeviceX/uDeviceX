struct PidVCont;
struct Coords;
struct Config;
struct float3;
struct int3;
struct Particle;

// tag::mem[]
void vcont_ini(MPI_Comm comm, int3 L, /**/ PidVCont **c);
void vcont_fin(/**/ PidVCont *cont);
// end::mem[]

// tag::set[]
void vcont_set_params(float factor, float Kp, float Ki, float Kd, /**/ PidVCont *c); // <1>
void vcont_set_target(float3 vtarget, /**/ PidVCont *c);  // <2>
void vcont_set_cart(/**/ PidVCont *c);  // <3>
void vcont_set_radial(/**/ PidVCont *c);  // <4>
// end::set[]

// tag::cnf[]
void vcont_set_conf(const Config *cfg, /**/ PidVCont *c);
// end::cnf[]

// tag::int[]
void   vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c); // <1>
float3 vcont_adjustF(/**/ PidVCont *c); // <2>
void   vcont_log(const PidVCont *c);    // <3>
// end::int[]

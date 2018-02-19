struct PidVCont;
struct Coords;

// tag::mem[]
void vcont_ini(MPI_Comm comm, int3 L, /**/ PidVCont **c);
void vcont_fin(/**/ PidVCont *cont);
// end::mem[]

void vcont_set_params(float factor, float Kp, float Ki, float Kd, /**/ PidVCont *c);
void vcont_set_target(float3 vtarget, /**/ PidVCont *c);

// tag::ini[]
void vcon_set_cart(/**/ PidVCont *cont);
void vcon_set_radial(/**/ PidVCont *cont);
// end::ini[]

// tag::int[]
void   vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c); // <1>
float3 vcont_adjustF(/**/ PidVCont *c); // <2>
void   vcont_log(const PidVCont *c);    // <3>
// end::int[]

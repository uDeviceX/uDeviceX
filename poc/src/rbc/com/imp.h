struct RbcCom;
struct float3;
struct Particle;

// tag::interface[]
void rbc_com_ini(int nv, int nm_max, /**/ RbcCom**);
void rbc_com_fin(/**/ RbcCom*);
void rbc_com_apply(RbcCom*, int nm, const Particle*, /**/ float3 **rr, float3 **vv);
// end::interface[]

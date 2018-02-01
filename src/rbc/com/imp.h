struct RbcCom;
void rbc_com_ini(int nv, int max_cell, /**/ RbcCom **);
void rbc_com_fin(/**/ RbcCom*);
void rbc_com_compute(RbcCom*, int nm, const Particle*, /**/ float3 **rr, float3 **vv);

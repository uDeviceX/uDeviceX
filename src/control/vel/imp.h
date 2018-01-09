struct PidVCont;

void vcont_ini(MPI_Comm comm, int3 L, float3 vtarget, float factor, /**/ PidVCont **c);
void vcont_fin(/**/ PidVCont *cont);

void vcon_set_cart(/**/ PidVCont *cont);
void vcon_set_radial(/**/ PidVCont *cont);

void   vcont_sample(Coords coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c);
float3 vcont_adjustF(/**/ PidVCont *c);
void   vcont_log(const PidVCont *c);

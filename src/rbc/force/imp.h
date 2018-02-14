struct RbcParams;
struct RbcForce;
struct Force;
void rbc_force_ini(int nv, int seed, RbcForce**);
void rbc_force_fin(RbcForce*);
void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*);
void rbc_force_stat(/**/ float *pArea, float *pVolume);

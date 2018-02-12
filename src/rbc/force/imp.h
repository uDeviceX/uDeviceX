struct RbcParams;
struct RbcForce;
struct Force;
void rbc_force_ini(const RbcQuants*, RbcForce**);
void rbc_force_fin(RbcForce*);
void rbc_force_apply(RbcForce*, const RbcParams *p, float dt, const RbcQuants*, /**/ Force *ff);
void rbc_force_stat(/**/ float *pArea, float *pVolume);

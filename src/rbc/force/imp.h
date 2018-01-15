struct RbcParams;
struct RbcForce { RbcRnd *rnd; };
void rbc_force_gen(const RbcQuants q, RbcForce *t);
void rbc_force_fin(RbcForce *t);
void rbc_force_apply(const RbcQuants q, const RbcForce t, const RbcParams *p, /**/ Force *ff);
void rbc_force_stat(/**/ float *pArea, float *pVolume);

/* TODO remove */
void ini_rbc_params(RbcParams *p);

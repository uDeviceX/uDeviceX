struct RbcParams;
struct RbcRnd;
struct RbcForce { RbcRnd *rnd; };
struct Force;
void rbc_force_gen(const RbcQuants q, RbcForce *t);
void rbc_force_fin(RbcForce *t);
void rbc_force_apply(float dt0, const RbcQuants q, const RbcForce t, const RbcParams *p, /**/ Force *ff);
void rbc_force_stat(/**/ float *pArea, float *pVolume);

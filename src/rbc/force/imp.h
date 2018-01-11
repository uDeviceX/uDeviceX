namespace rbc { namespace force {
struct RbcForce { RbcRnd *rnd; };
void rbcforce_gen(const RbcQuants q, RbcForce *t);
void rbcforce_fin(RbcForce *t);
void rbcforce_apply(const RbcQuants q, const RbcForce t, /**/ Force *ff);
void rbcforce_stat(/**/ float *pArea, float *pVolume);
}} /* namespace */

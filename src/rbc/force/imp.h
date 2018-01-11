namespace rbc { namespace force {
struct RbcForce { rbc::rnd::D *rnd; };
void gen_ticket(const RbcQuants q, RbcForce *t);
void fin_ticket(RbcForce *t);
void apply(const RbcQuants q, const RbcForce t, /**/ Force *ff);
void stat(/**/ float *pArea, float *pVolume);
}} /* namespace */

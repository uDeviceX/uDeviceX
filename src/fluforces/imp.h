struct FluForcesBulk;
struct FluForcesHalo;

void ini(int maxp, /**/ FluForcesBulk **b);
void fin(/**/ FluForcesBulk *b);

void prepare(int n, const Cloud *c, /**/ FluForcesBulk *b);
void bulk_forces(int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff);


void ini(MPI_Comm cart, /**/ FluForcesHalo **hd);
void fin(/**/ FluForcesHalo *h);

void prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags, /**/ FluForcesHalo *h);
void halo_forces(const FluForcesHalo *h, /**/ Force *ff);

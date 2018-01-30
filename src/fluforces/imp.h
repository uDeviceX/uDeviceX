struct FluForcesBulk;
struct FluForcesHalo;

void fluforces_bulk_ini(int maxp, /**/ FluForcesBulk **b);
void fluforces_bulk_fin(/**/ FluForcesBulk *b);

void fluforces_bulk_prepare(int n, const Cloud *c, /**/ FluForcesBulk *b);
void fluforces_bulk_apply(int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff);


void fluforces_halo_ini(MPI_Comm cart, int3 L, /**/ FluForcesHalo **hd);
void fluforces_halo_fin(/**/ FluForcesHalo *h);

void fluforces_halo_prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags, /**/ FluForcesHalo *h);
void fluforces_halo_apply(const FluForcesHalo *h, /**/ Force *ff);

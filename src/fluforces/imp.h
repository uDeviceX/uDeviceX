struct FluForcesBulk;
struct HaloData;

void ini(int maxp, /**/ FluForcesBulk **b);
void fin(/**/ FluForcesBulk *b);

void prepare(int n, const Cloud *c, /**/ FluForcesBulk *b);
void bulk_forces(int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff);


void ini(MPI_Comm cart, /**/ HaloData **hd);
void fin(/**/ HaloData *h);

void prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags, /**/ HaloData *h);
void halo_forces(const HaloData *h, /**/ Force *ff);

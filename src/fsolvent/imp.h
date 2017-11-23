struct BulkData;
struct HaloData;

void ini(int maxp, /**/ BulkData **b);
void fin(/**/ BulkData *b);

void prepare(int n, const Cloud *c, /**/ BulkData *b);
void bulk_forces(int n, const BulkData *b, const int *start, const int *count, /**/ Force *ff);


void prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags, /**/ HaloData *h);

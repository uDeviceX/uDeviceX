struct PairParams;
struct int3;
struct FoArray;

void fhalo_apply(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray);

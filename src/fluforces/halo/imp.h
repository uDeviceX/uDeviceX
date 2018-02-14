struct PairParams;
struct int3;

void fhalo_apply      (const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff);
void fhalo_apply_color(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff);

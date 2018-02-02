struct PairParams;
struct int3;

namespace hforces {

void fhalo      (const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff);
void fhalo_color(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff);

} /* namespace */

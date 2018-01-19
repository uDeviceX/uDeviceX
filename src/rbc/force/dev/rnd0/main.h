struct Rnd0 { };

static __device__ void edg_rnd(Shape, int, float*, int, Rnd0*) { }
static __device__ float3 frnd(RbcParams_v, float3, float3, const Rnd0) { return make_float3(0, 0, 0); }
